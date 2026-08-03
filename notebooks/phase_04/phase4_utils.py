"""
Phase 4 -- From mechanism to function: is the query-driven sink causally
responsible for model performance?

Phases 1-3 established that the sink exists, is query-driven, and that a full
query transplant reproduces the attention pattern. Phase 4 tests whether that
pattern *matters for the output*, via five experiments that share one editable
attention path and one teacher-forced functional harness:

    Exp 1  sink necessity       : remove BOS attention + renormalize
    Exp 2  control ablation      : remove a matched non-sink key (specificity)
    Exp 3  dose-response         : scale BOS attention by gamma, renormalize
    Exp 4  query intervention    : suppress / project-out the sink direction in q
    Exp 5  streaming             : windowed attention +/- sink retention

Design decisions locked for the phase:
  * functional metrics (NLL/KL/top-1) are reported on a held-out corpus; the
    frozen benchmark is used only for mechanistic comparability with 1-3;
  * the sink-head set H* and matched non-sink set H-circle are FROZEN inputs
    (from Phase 2), never recomputed on the eval corpus.

Intervention mechanism: with attn_implementation='eager', Qwen3's forward calls
the module-level ``eager_attention_forward``. We temporarily replace it with an
edition-aware version that (a) can project the sink direction out of the query
before scores, and (b) can remove/scale/mask attention-weight columns after
softmax and renormalize. The identity "post-softmax zero+renorm == pre-softmax
-inf mask" is used as a validation target. All edits restore on context exit.

Architecture facts used: no Q/K/V bias; q_norm/k_norm are per-head-shared RMSNorm;
RoPE is identity at position 0, so the sink key k_0 is unrotated; head h reads KV
group h // n_rep.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

import matplotlib
try:
    import IPython
    _IN_IPY = IPython.get_ipython() is not None
except Exception:
    _IN_IPY = False
if not _IN_IPY:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase3_utils as U
from transformers.models.qwen3 import modeling_qwen3 as _mq

PathLike = Union[str, Path]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ==========================================================================
# 0. topology helpers
# ==========================================================================

def model_topology(model: Any) -> Dict[str, int]:
    c = model.config
    Hq = int(c.num_attention_heads); Hkv = int(c.num_key_value_heads)
    D = int(getattr(c, "head_dim", c.hidden_size // Hq))
    return {"n_layers": int(c.num_hidden_layers), "n_heads": Hq, "n_kv": Hkv,
            "n_rep": Hq // Hkv, "head_dim": D}


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, hkv, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, hkv, n_rep, s, d).reshape(b, hkv * n_rep, s, d)


# ==========================================================================
# 1. Attention-weight editor
#
# One EditConfig describes an intervention; the context manager installs an
# edition-aware eager_attention_forward that reads the active config. Heads are
# selected per layer via `heads`: a dict {layer_idx: set(heads)} or the string
# 'all'. `pre_softmax_mask` toggles the equivalence-check path for removal.
# ==========================================================================

@dataclass
class EditConfig:
    kind: str                       # 'none'|'remove'|'scale'|'project_bos'|'window'
    heads: Union[str, Dict[int, set]] = "all"
    key_index: int = 0              # column to remove/scale (0 = BOS/sink)
    key_index_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None
    gamma: float = 1.0              # scale factor for 'scale'
    window: int = 0                 # sliding window W for 'window'
    sink_keep: int = 0              # # of leading sink tokens retained for 'window'
    pre_softmax_mask: bool = False  # remove via -inf pre-softmax (equivalence check)
    min_query_pos: int = 1          # never edit rows below this (row 0 always safe)

    def applies(self, layer_idx: Optional[int]) -> bool:
        if self.kind == "none" or layer_idx is None:
            return False
        if self.heads == "all":
            return True
        return layer_idx in self.heads and len(self.heads[layer_idx]) > 0

    def head_mask(self, layer_idx: int, Hq: int, device) -> torch.Tensor:
        """Boolean [Hq] mask of heads to edit at this layer."""
        if self.heads == "all":
            return torch.ones(Hq, dtype=torch.bool, device=device)
        m = torch.zeros(Hq, dtype=torch.bool, device=device)
        for h in self.heads.get(layer_idx, ()):
            m[h] = True
        return m


# module-global holding the active edit (set only inside the context manager)
_ACTIVE: Optional[EditConfig] = None
_ORIG_EAGER = _mq.eager_attention_forward


def _project_out_bos(query: torch.Tensor, key: torch.Tensor,
                     head_mask: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Remove, per edited head, the component of the (post-RoPE) query along that
    head's sink key k_0 = key[:, g, 0, :] (RoPE is identity at position 0). Sets the
    BOS logit to exactly 0 for edited heads while touching non-sink logits only
    through k_0's overlap with other keys."""
    B, Hq, T, D = query.shape
    q = query.clone()
    for h in range(Hq):
        if not head_mask[h]:
            continue
        g = h // n_rep
        k0 = key[:, g, 0, :]                       # [B, D]
        k0n = k0 / k0.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        qh = q[:, h, :, :]                          # [B, T, D]
        coef = torch.einsum("btd,bd->bt", qh, k0n) # [B, T]
        q[:, h, :, :] = qh - coef.unsqueeze(-1) * k0n.unsqueeze(1)
    return q


def _edit_weights(aw: torch.Tensor, cfg: EditConfig, layer_idx: int) -> torch.Tensor:
    """Apply a post-softmax weight-space edit and renormalize. aw: [B,Hq,Tq,Tk]."""
    B, Hq, Tq, Tk = aw.shape
    hm = cfg.head_mask(layer_idx, Hq, aw.device)
    if not hm.any():
        return aw
    out = aw.clone()
    rows = torch.arange(Tq, device=aw.device)
    editable = rows >= cfg.min_query_pos          # [Tq] never touch row 0..min-1

    if cfg.kind in ("remove", "scale"):
        j = cfg.key_index
        factor = 0.0 if cfg.kind == "remove" else float(cfg.gamma)
        for h in range(Hq):
            if not hm[h]:
                continue
            block = out[:, h, :, :]                # [B,Tq,Tk]
            new = block.clone()
            new[:, editable, j] = block[:, editable, j] * factor
            out[:, h, :, :] = _renorm_rows(block, new, editable)

    elif cfg.kind == "window":
        allowed = _window_allowed(Tq, Tk, cfg.window, cfg.sink_keep, aw.device)  # [Tq,Tk] bool
        allowed_f = allowed.to(aw.dtype)
        for h in range(Hq):
            if not hm[h]:
                continue
            block = out[:, h, :, :]
            new = block * allowed_f.unsqueeze(0)
            out[:, h, :, :] = _renorm_rows(block, new, editable)
    return out


def _renorm_rows(orig: torch.Tensor, edited: torch.Tensor,
                 editable: torch.Tensor) -> torch.Tensor:
    """Renormalize edited rows to sum 1; rows that became all-zero (no surviving
    key) revert to the original row. Non-editable rows keep the original."""
    s = edited.sum(dim=-1, keepdim=True)            # [B,Tq,1]
    safe = s.squeeze(-1) > 0                         # [B,Tq]
    renormed = edited / s.clamp_min(1e-30)
    ed = editable.view(1, -1, 1)
    keep_orig = (~editable).view(1, -1) | (~safe)   # [B?,Tq] broadcast
    out = torch.where(ed & safe.unsqueeze(-1), renormed, orig)
    return out


def _window_allowed(Tq: int, Tk: int, W: int, sink_keep: int, device) -> torch.Tensor:
    """[Tq,Tk] boolean: query i may see keys {0..sink_keep-1} U {i-W+1..i}, causal."""
    qi = torch.arange(Tq, device=device).view(-1, 1)
    kj = torch.arange(Tk, device=device).view(1, -1)
    causal = kj <= qi
    in_window = kj > (qi - W)
    is_sink = kj < sink_keep
    return causal & (in_window | is_sink)


def _edited_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Drop-in replacement for eager_attention_forward that applies _ACTIVE edits."""
    cfg = _ACTIVE
    L = getattr(module, "layer_idx", None)
    n_rep = module.num_key_value_groups

    if cfg is not None and cfg.kind == "project_bos" and cfg.applies(L):
        hm = cfg.head_mask(L, query.shape[1], query.device)
        query = _project_out_bos(query, key, hm, n_rep)

    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    aw = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : key_states.shape[-2]]

    # optional pre-softmax removal (equivalence-check path)
    if (cfg is not None and cfg.kind == "remove" and cfg.pre_softmax_mask and cfg.applies(L)):
        hm = cfg.head_mask(L, aw.shape[1], aw.device)
        Tq = aw.shape[2]
        editable = torch.arange(Tq, device=aw.device) >= cfg.min_query_pos
        for h in range(aw.shape[1]):
            if hm[h]:
                aw[:, h, editable, cfg.key_index] = float("-inf")

    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)

    if (cfg is not None and cfg.kind in ("remove", "scale", "window")
            and not cfg.pre_softmax_mask and cfg.applies(L)):
        aw = _edit_weights(aw, cfg, L)

    attn_output = torch.matmul(aw, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, aw


@contextmanager
def attention_edit(model: Any, cfg: Optional[EditConfig]):
    """Install the edition-aware attention for the duration of the context. Requires
    the model to use attn_implementation='eager'. Restores on exit."""
    global _ACTIVE, _ORIG_EAGER
    prev = _ACTIVE
    _ACTIVE = cfg
    _ORIG_EAGER = _mq.eager_attention_forward
    _mq.eager_attention_forward = _edited_eager
    # also patch the interface registry if the model resolves through it
    try:
        from transformers.masking_utils import ALL_ATTENTION_FUNCTIONS  # noqa
    except Exception:
        ALL_ATTENTION_FUNCTIONS = None
    try:
        yield
    finally:
        _mq.eager_attention_forward = _ORIG_EAGER
        _ACTIVE = prev


# ==========================================================================
# 2. Functional metrics (teacher forcing)
# ==========================================================================

@dataclass
class ForwardResult:
    nll: torch.Tensor        # [B, T-1] per query position (predicting t+1 from <=t)
    logits: torch.Tensor     # [B, T, V]
    input_ids: torch.Tensor


def teacher_forced(model: Any, input_ids: torch.Tensor) -> ForwardResult:
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = input_ids[:, 1:]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    return ForwardResult(nll=nll, logits=logits, input_ids=input_ids)


def run_edited(model: Any, input_ids: torch.Tensor,
               cfg: Optional[EditConfig]) -> ForwardResult:
    with attention_edit(model, cfg):
        return teacher_forced(model, input_ids)


def kl_top1(base_logits: torch.Tensor, int_logits: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position KL(base||int) over the next-token distribution, and a boolean
    top-1 agreement mask. Inputs [B,T,V]; outputs [B,T]."""
    lpb = torch.log_softmax(base_logits.float(), dim=-1)
    lpi = torch.log_softmax(int_logits.float(), dim=-1)
    kl = (lpb.exp() * (lpb - lpi)).sum(dim=-1)
    agree = base_logits.argmax(-1) == int_logits.argmax(-1)
    return kl, agree


def _positional_frame(base: ForwardResult, intr: ForwardResult, *, condition: str,
                      seq_index: int, min_query_pos: int, extra: Dict[str, Any]
                      ) -> pd.DataFrame:
    """One row per query position with paired baseline/intervention functional metrics."""
    kl, agree = kl_top1(base.logits, intr.logits)     # [B,T]
    T = base.nll.shape[1]                             # query positions 0..T-1 (predict t+1)
    pos = torch.arange(T)
    keep = pos >= min_query_pos
    d = {
        "seq": seq_index,
        "token": pos[keep].numpy(),
        "nll_base": base.nll[0, keep].float().cpu().numpy(),
        "nll_int": intr.nll[0, keep].float().cpu().numpy(),
        "delta_nll": (intr.nll[0, keep] - base.nll[0, keep]).float().cpu().numpy(),
        "kl": kl[0, :T][keep].float().cpu().numpy(),
        "top1_agree": agree[0, :T][keep].cpu().numpy().astype(float),
    }
    d.update({k: v for k, v in extra.items()})
    d["condition"] = condition
    return pd.DataFrame(d)


# ==========================================================================
# 3. Mechanistic side-channel (baseline sink mass, ||v0||, residual drift)
# ==========================================================================

def baseline_sink_mass(model: Any, input_ids: torch.Tensor,
                       layers: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Per (layer, head) mean baseline sink mass, via the Phase-3 capture."""
    cfg = U.CaptureConfig(layers=list(layers)) if layers is not None else U.CaptureConfig()
    with U.capture_attention(model, cfg) as rec:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    cap = rec.finalize(input_ids=input_ids, model=model)
    qm = U.compute_query_metrics(cap)
    tbl = U.build_metric_table(qm, seq_index=0, min_query_pos=4)
    df = pd.DataFrame({n: np.asarray(tbl[n]) for n in tbl.names})
    return (df.groupby(["layer", "head"], as_index=False)["bos_prob"]
              .mean().rename(columns={"bos_prob": "sink_mass"}))


def value_sink_norms(model: Any, input_ids: torch.Tensor) -> pd.DataFrame:
    """||v_0|| (BOS value norm) per (layer, KV group), for the null-attention test."""
    rows = []
    handles = []
    store: Dict[int, torch.Tensor] = {}

    def mk(L):
        def hook(mod, inp, out):
            top = model_topology(model)
            B, T, HD = out.shape
            v = out.view(B, T, top["n_kv"], top["head_dim"])
            store[L] = v[:, 0, :, :].detach().norm(dim=-1)[0].cpu()  # [n_kv]
        return hook

    for L, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    for L, v in store.items():
        for g in range(v.shape[0]):
            rows.append({"layer": L, "kv_group": g, "v0_norm": float(v[g])})
    return pd.DataFrame(rows)


# ==========================================================================
# 4. Scoping: build head-selection dicts from a frozen sink-head list
# ==========================================================================

def heads_dict(pairs: Sequence[Tuple[int, int]]) -> Dict[int, set]:
    d: Dict[int, set] = {}
    for (L, h) in pairs:
        d.setdefault(int(L), set()).add(int(h))
    return d


def matched_nonsink_set(sink_pairs: Sequence[Tuple[int, int]],
                        sink_mass_df: pd.DataFrame, seed: int = 0
                        ) -> List[Tuple[int, int]]:
    """Pick a non-sink control set of the same size, matched on layer-depth
    distribution, from the lowest-sink heads at each layer. Deterministic given
    the frozen sink-mass table; intended to be computed ONCE and frozen."""
    rng = np.random.default_rng(seed)
    sink_by_layer: Dict[int, int] = {}
    sink_set = set((int(L), int(h)) for (L, h) in sink_pairs)
    for (L, h) in sink_pairs:
        sink_by_layer[int(L)] = sink_by_layer.get(int(L), 0) + 1
    out: List[Tuple[int, int]] = []
    for L, n in sink_by_layer.items():
        cands = (sink_mass_df[sink_mass_df["layer"] == L]
                 .sort_values("sink_mass"))
        cands = [(int(r.layer), int(r.head)) for r in cands.itertuples()
                 if (int(r.layer), int(r.head)) not in sink_set]
        out.extend(cands[:n])
    return out


# ==========================================================================
# 5. Experiments
# ==========================================================================

def _as_batches(prompts: Sequence[Any]) -> List[torch.Tensor]:
    out = []
    for p in prompts:
        ids = p if isinstance(p, torch.Tensor) else torch.as_tensor(p)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out.append(ids)
    return out


def _scope_configs(kind: str, sink_pairs, nonsink_pairs, **kw) -> Dict[str, EditConfig]:
    return {
        "global": EditConfig(kind=kind, heads="all", **kw),
        "sink": EditConfig(kind=kind, heads=heads_dict(sink_pairs), **kw),
        "nonsink": EditConfig(kind=kind, heads=heads_dict(nonsink_pairs), **kw),
    }


def experiment_necessity(model, prompts, sink_pairs, nonsink_pairs,
                         min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Exp 1. Remove BOS attention (key 0) + renormalize under each scope; measure
    the functional cost vs baseline."""
    batches = _as_batches(prompts)
    scopes = _scope_configs("remove", sink_pairs, nonsink_pairs,
                            key_index=0, min_query_pos=min_query_pos)
    frames = []
    for si, ids in enumerate(batches):
        base = teacher_forced(model, ids)
        for name, cfg in scopes.items():
            if progress and si == 0:
                print(f"    [Exp1] scope={name}", flush=True)
            intr = run_edited(model, ids, cfg)
            frames.append(_positional_frame(base, intr, condition=name, seq_index=si,
                                             min_query_pos=min_query_pos,
                                             extra={"experiment": "necessity"}))
    return pd.concat(frames, ignore_index=True)


def experiment_control(model, prompts, sink_pairs, nonsink_pairs,
                       min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Exp 2. Remove a matched non-sink key (fixed position 1) as a specificity
    control, alongside BOS removal, at the sink scope. (Mass-matched selection is
    available via key_index_fn; fixed-position is the reproducible default.)"""
    batches = _as_batches(prompts)
    sink_heads = heads_dict(sink_pairs)
    conds = {
        "remove_bos": EditConfig("remove", sink_heads, key_index=0, min_query_pos=min_query_pos),
        "remove_pos1": EditConfig("remove", sink_heads, key_index=1, min_query_pos=min_query_pos),
    }
    frames = []
    for si, ids in enumerate(batches):
        base = teacher_forced(model, ids)
        for name, cfg in conds.items():
            if progress and si == 0:
                print(f"    [Exp2] {name}", flush=True)
            intr = run_edited(model, ids, cfg)
            frames.append(_positional_frame(base, intr, condition=name, seq_index=si,
                                             min_query_pos=min_query_pos,
                                             extra={"experiment": "control"}))
    return pd.concat(frames, ignore_index=True)


def experiment_dose(model, prompts, sink_pairs,
                    gammas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                    min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Exp 3. Scale BOS attention by gamma (then renormalize) at the sink scope;
    gamma=1 recovers baseline, gamma=0 reduces to Exp 1."""
    batches = _as_batches(prompts)
    sink_heads = heads_dict(sink_pairs)
    frames = []
    for si, ids in enumerate(batches):
        base = teacher_forced(model, ids)
        for g in gammas:
            if progress and si == 0:
                print(f"    [Exp3] gamma={g}", flush=True)
            cfg = EditConfig("scale", sink_heads, key_index=0, gamma=float(g),
                             min_query_pos=min_query_pos)
            intr = run_edited(model, ids, cfg)
            fr = _positional_frame(base, intr, condition=f"gamma_{g:.2f}", seq_index=si,
                                   min_query_pos=min_query_pos,
                                   extra={"experiment": "dose", "gamma": float(g)})
            frames.append(fr)
    return pd.concat(frames, ignore_index=True)


def experiment_query(model, prompts, sink_pairs,
                     min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Exp 4. Query-side sink interventions with FUNCTIONAL read-outs: project the
    sink direction out of the query (surgical) and suppress the query (upper bound),
    at the sink scope."""
    batches = _as_batches(prompts)
    sink_heads = heads_dict(sink_pairs)
    conds = {
        "project_bos": EditConfig("project_bos", sink_heads, min_query_pos=min_query_pos),
    }
    frames = []
    for si, ids in enumerate(batches):
        base = teacher_forced(model, ids)
        for name, cfg in conds.items():
            if progress and si == 0:
                print(f"    [Exp4] {name}", flush=True)
            intr = run_edited(model, ids, cfg)
            frames.append(_positional_frame(base, intr, condition=name, seq_index=si,
                                             min_query_pos=min_query_pos,
                                             extra={"experiment": "query"}))
    return pd.concat(frames, ignore_index=True)


def experiment_streaming(model, prompts, sink_pairs,
                         window: int = 64, sink_keeps: Sequence[int] = (0, 1, 4),
                         min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Exp 5 (masking approximation). Restrict attention to a sliding window of size
    W, retaining `sink_keep` leading sink tokens, GLOBALLY (all heads). This measures
    the per-position NLL cost of windowed attention with/without sink retention.

    NOTE: this is the dense-mask proxy for streaming -- positions/RoPE are the
    original ones (no re-basing). True KV eviction with re-based positions is a
    separate, heavier implementation and is flagged as an extension.
    """
    batches = _as_batches(prompts)
    frames = []
    for si, ids in enumerate(batches):
        base = teacher_forced(model, ids)
        for k in sink_keeps:
            if progress and si == 0:
                print(f"    [Exp5] window={window} sink_keep={k}", flush=True)
            cfg = EditConfig("window", heads="all", window=window, sink_keep=int(k),
                             min_query_pos=min_query_pos)
            intr = run_edited(model, ids, cfg)
            fr = _positional_frame(base, intr, condition=f"W{window}_sink{k}", seq_index=si,
                                   min_query_pos=min_query_pos,
                                   extra={"experiment": "streaming", "window": window,
                                          "sink_keep": int(k)})
            frames.append(fr)
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# 6. Aggregation
# ==========================================================================

def summarize(df: pd.DataFrame, by: Sequence[str] = ("condition",)) -> pd.DataFrame:
    """Token -> sequence -> overall pooling of the functional metrics, with a
    sequence-bootstrap CI on delta_nll."""
    by = list(by)
    l2 = df.groupby(by + ["seq"], as_index=False)[
        ["delta_nll", "kl", "top1_agree", "nll_base", "nll_int"]].mean()
    rows = []
    for key, grp in l2.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        vals = grp["delta_nll"].values
        boot = _bootstrap_ci(vals)
        rows.append({**dict(zip(by, key)),
                     "delta_nll": float(vals.mean()),
                     "delta_nll_lo": boot[0], "delta_nll_hi": boot[1],
                     "kl": float(grp["kl"].mean()),
                     "top1_agree": float(grp["top1_agree"].mean()),
                     "n_seq": len(grp)})
    return pd.DataFrame(rows)


def _bootstrap_ci(x: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ==========================================================================
# 7. Plots
# ==========================================================================

def plot_scope_bars(summary: pd.DataFrame, out_path: PathLike, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    s = summary.copy()
    x = np.arange(len(s))
    err = np.vstack([s["delta_nll"] - s["delta_nll_lo"], s["delta_nll_hi"] - s["delta_nll"]])
    ax.bar(x, s["delta_nll"], yerr=err, capsize=4, color="#4C72B0", alpha=0.85)
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(s["condition"], rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta$NLL (nats)"); ax.set_title("Functional cost by scope", loc="left")
    return _save(fig, out_path, show)


def plot_dose(df: pd.DataFrame, out_path: PathLike, show: bool = False) -> Path:
    l2 = df.groupby(["gamma", "seq"], as_index=False)[["delta_nll", "kl", "top1_agree"]].mean()
    agg = l2.groupby("gamma", as_index=False).agg(
        dnll=("delta_nll", "mean"), dnll_s=("delta_nll", "std"),
        kl=("kl", "mean"), top1=("top1_agree", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].errorbar(agg["gamma"], agg["dnll"], yerr=agg["dnll_s"].fillna(0), fmt="-o", capsize=3)
    ax[0].set_xlabel(r"$\gamma$ (retained sink attention)"); ax[0].set_ylabel(r"$\Delta$NLL")
    ax[0].set_title("(a) dose-response", loc="left"); ax[0].invert_xaxis()
    ax[1].plot(agg["gamma"], agg["kl"], "-o", label="KL")
    ax[1].plot(agg["gamma"], agg["top1"], "-s", label="top-1 agree")
    ax[1].set_xlabel(r"$\gamma$"); ax[1].set_title("(b) KL & top-1 vs dose", loc="left")
    ax[1].legend(); ax[1].invert_xaxis()
    return _save(fig, out_path, show)


def plot_position_curve(df: pd.DataFrame, value: str, out_path: PathLike,
                        hue: str = "condition", show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name, grp in df.groupby(hue):
        prof = grp.groupby("token")[value].mean()
        ax.plot(prof.index, prof.values, lw=1.4, label=str(name))
    ax.set_xlabel("query position"); ax.set_ylabel(value)
    ax.set_title(f"{value} vs position", loc="left"); ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_path, show)


def plot_streaming(df: pd.DataFrame, out_path: PathLike, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for k, grp in df.groupby("sink_keep"):
        prof = grp.groupby("token")["nll_int"].mean()
        ax.plot(prof.index, prof.values, lw=1.3, label=f"sink_keep={k}")
    base = df.groupby("token")["nll_base"].mean()
    ax.plot(base.index, base.values, "k--", lw=1, label="full-context baseline")
    ax.set_xlabel("position"); ax.set_ylabel("NLL (nats)")
    ax.set_title("Streaming NLL vs position (windowed attention)", loc="left")
    ax.legend(fontsize=8)
    return _save(fig, out_path, show)


def _save(fig, out_path: PathLike, show: bool) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.show() if show else plt.close(fig)
    return out_path


# ==========================================================================
# 8. Frozen Phase-2 head sets + rank-transfer validation (Spearman)
# ==========================================================================

def load_phase2_matrix(path: PathLike) -> pd.DataFrame:
    """Load the Phase-2 layer_head_matrix.csv (rows layerN, cols headM) into a long
    (layer, head, sink_mass) frame -- the frozen source for H*."""
    M = pd.read_csv(path, index_col=0)
    M.index = [int(str(x).replace("layer", "")) for x in M.index]
    M.columns = [int(str(c).replace("head", "")) for c in M.columns]
    long = M.stack().reset_index()
    long.columns = ["layer", "head", "sink_mass"]
    return long


def select_sink_heads(joint: pd.DataFrame, threshold: float) -> List[Tuple[int, int]]:
    """Frozen H*: all (layer, head) cells with sink_mass >= threshold, strongest first."""
    sel = joint[joint["sink_mass"] >= threshold].sort_values("sink_mass", ascending=False)
    return [(int(r.layer), int(r.head)) for r in sel.itertuples()]


def sink_mass_table(model: Any, batches: Sequence[torch.Tensor],
                    layers: Optional[Sequence[int]] = None,
                    max_batches: Optional[int] = None) -> pd.DataFrame:
    """Mean baseline sink mass per (layer, head) over a set of input batches
    (e.g. held-out WikiText windows), via the Phase-3 capture."""
    frames = []
    for i, ids in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        frames.append(baseline_sink_mass(model, ids, layers=layers))
    return (pd.concat(frames).groupby(["layer", "head"], as_index=False)["sink_mass"].mean())


def spearman_rank_agreement(ref: pd.DataFrame, test: pd.DataFrame,
                            col: str = "sink_mass") -> Tuple[float, pd.DataFrame]:
    """Spearman rho between two per-(layer, head) sink tables. Spearman == Pearson of
    ranks, so no scipy dependency. Returns (rho, merged frame with rank columns)."""
    m = (ref.rename(columns={col: "sink_ref"})
            .merge(test.rename(columns={col: "sink_test"}), on=["layer", "head"]))
    if len(m) < 3:
        return float("nan"), m
    m["rank_ref"] = m["sink_ref"].rank()
    m["rank_test"] = m["sink_test"].rank()
    rho = float(np.corrcoef(m["rank_ref"], m["rank_test"])[0, 1])
    return rho, m


def plot_rank_agreement(merged: pd.DataFrame, rho: float, out_path: PathLike,
                        sink_pairs: Optional[Sequence[Tuple[int, int]]] = None,
                        show: bool = False) -> Path:
    """Scatter of Phase-2 vs held-out sink mass per (layer, head), Spearman annotated;
    H* cells highlighted."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax[0].scatter(merged["sink_ref"], merged["sink_test"], s=14, alpha=0.5, color="#4C72B0")
    if sink_pairs:
        sset = set((int(a), int(b)) for (a, b) in sink_pairs)
        key = list(zip(merged["layer"].astype(int), merged["head"].astype(int)))
        hi = merged[[k in sset for k in key]]
        ax[0].scatter(hi["sink_ref"], hi["sink_test"], s=26, color="#C44E52",
                      label=r"$\mathcal{H}^\star$")
        ax[0].legend(fontsize=8)
    lim = [0, 1]
    ax[0].plot(lim, lim, "--", color="0.6", lw=1)
    ax[0].set_xlabel("Phase-2 sink score"); ax[0].set_ylabel("held-out (WikiText) sink score")
    ax[0].set_title("(a) sink mass per (layer, head)", loc="left")
    ax[1].scatter(merged["rank_ref"], merged["rank_test"], s=12, alpha=0.5, color="#55A868")
    ax[1].set_xlabel("Phase-2 rank"); ax[1].set_ylabel("held-out rank")
    ax[1].set_title(f"(b) rank agreement · Spearman $\\rho$ = {rho:.3f}", loc="left")
    fig.suptitle("Transfer validation: does the frozen Phase-2 sink structure hold on held-out text?",
                 fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# 9. Memory-light sink-mass recorder (for long windows)
#
# baseline_sink_mass reuses the full Phase-3 capture, which on 512-token windows
# materializes the whole [L,H,T,T] attention tensor plus float64 metric temporaries
# -- several GB per window. For the transfer-validation ranking we only need the
# mean BOS attention per (layer, head), so we hook the eager attention, reduce to
# the sink column immediately, and discard the matrix. Peak memory ~ one layer's
# [B,H,T,T] at a time.
# ==========================================================================

_SINK_REC: Optional[Dict[str, Any]] = None


def _recording_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    n_rep = module.num_key_value_groups
    ks = _repeat_kv(key, n_rep); vs = _repeat_kv(value, n_rep)
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    rec = _SINK_REC
    if rec is not None:
        L = getattr(module, "layer_idx", None)
        mp = rec["min_pos"]
        if aw.shape[2] > mp:
            bos = aw[:, :, mp:, 0].mean(dim=(0, 2)).detach().float().cpu()   # [Hq]
            rec["acc"][L] = rec["acc"].get(L, 0.0) + bos
            rec["count"][L] = rec["count"].get(L, 0) + 1
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


@contextmanager
def _record_sink(min_pos: int):
    global _SINK_REC, _ORIG_EAGER
    prev = _SINK_REC
    _SINK_REC = {"acc": {}, "count": {}, "min_pos": min_pos}
    saved = _mq.eager_attention_forward
    _mq.eager_attention_forward = _recording_eager
    try:
        yield _SINK_REC
    finally:
        _mq.eager_attention_forward = saved
        _SINK_REC = prev


def sink_mass_fast(model: Any, batches: Sequence[torch.Tensor],
                   min_query_pos: int = 4, max_batches: Optional[int] = None) -> pd.DataFrame:
    """Mean baseline sink mass per (layer, head) over `batches`, memory-light:
    reduces each layer's attention to the BOS column on the fly. Matches
    baseline_sink_mass but is safe on long (512-token) windows."""
    with _record_sink(min_query_pos) as rec:
        for i, ids in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            with torch.no_grad():
                model(ids, use_cache=False)
    rows = []
    for L, s in rec["acc"].items():
        mean = s / rec["count"][L]                     # [Hq]
        for h in range(len(mean)):
            rows.append({"layer": int(L), "head": int(h), "sink_mass": float(mean[h])})
    return pd.DataFrame(rows)
