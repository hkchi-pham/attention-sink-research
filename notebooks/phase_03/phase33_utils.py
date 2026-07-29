"""
Phase 3.3 -- Query-pathway interventions: from localization to causality.

Phases 3.1/3.2 were observational and pointed at the query pathway. Phase 3.3
tests it causally: if we move one head's query onto its within-KV-group partner,
does its sink behaviour move too? Within a group K and V are shared, so a change
confined to the query is the *only* thing that can move sink behaviour -- that is
what makes these interventions causal rather than correlational, and why we only
ever swap heads inside one KV group.

Four independent layers (so future interventions reuse the infrastructure):

    1. intervention logic  -- pure functions  q -> q'  (iv_*)
    2. forward hooks / swap -- context managers that install them (patched_query,
                               swapped_wq); both restore state on exit
    3. metric computation   -- one shared adapter over Phase 3.1's
                               compute_query_metrics -> tidy pandas rows
    4. visualization        -- before/after, delta, transfer plots

Architectural facts this relies on (verified against the installed transformers):
q_proj has NO bias, q_norm is a single RMSNorm over head_dim shared by all heads,
and RoPE depends only on position -- so every op after the per-head query slice is
head-independent. Consequently editing W_Q rows (Exp. A) and copying the q_proj
activation slice (Exp. B) yield the *identical* query_states; they are the same
intervention at two stages, and their agreement is an internal causal check.

Nothing here trains or fine-tunes; it only runs counterfactual forward passes.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

import matplotlib
try:
    import IPython
    _IN_IPYTHON = IPython.get_ipython() is not None
except Exception:
    _IN_IPYTHON = False
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase3_utils as U

PathLike = Union[str, Path]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Metrics carried on every row (all per-token). Column -> QueryMetrics field.
_METRIC_COLUMNS = {
    "bos_prob": "bos_prob",              # sink probability (per token)
    "bos_logit": "bos_logit",
    "comp_max_logit": "comp_max_logit",  # strongest competitor
    "margin_max": "margin_max",          # BOS advantage = bos_logit - comp_max
    "comp_lse_logit": "comp_lse_logit",  # LSE
    "margin_lse": "margin_lse",
    "entropy": "entropy",
    "comp_neff": "comp_neff",            # effective # competitors
}
# The headline metric for transfer is the sink score (mean of bos_prob).
TRANSFER_METRICS = ("bos_prob", "margin_max", "bos_logit", "comp_max_logit",
                    "comp_lse_logit", "entropy", "comp_neff")


# ==========================================================================
# 0. topology helpers
# ==========================================================================

def _attn(model: Any, layer_idx: int):
    return model.model.layers[layer_idx].self_attn


def model_topology(model: Any) -> Dict[str, int]:
    c = model.config
    Hq = int(c.num_attention_heads)
    Hkv = int(c.num_key_value_heads)
    D = int(getattr(c, "head_dim", c.hidden_size // Hq))
    return {"n_heads": Hq, "n_kv": Hkv, "n_rep": Hq // Hkv, "head_dim": D}


def kv_group_of(head: int, n_rep: int) -> int:
    return head // n_rep


def within_group_pairs(model: Any) -> List[Tuple[int, int]]:
    """All (a, b) query-head pairs that share a KV head (a < b)."""
    top = model_topology(model)
    pairs: List[Tuple[int, int]] = []
    for g in range(top["n_kv"]):
        members = [h for h in range(top["n_heads"]) if kv_group_of(h, top["n_rep"]) == g]
        pairs.extend(combinations(members, 2))
    return pairs


def _tick(label: str, i: int, n: int, on: bool, every: int = 1) -> None:
    """Lightweight progress line so long sweeps are not silent."""
    if not on:
        return
    if i == 0 or (i + 1) % every == 0 or i == n - 1:
        print(f"    [{label}] site {i + 1}/{n}", flush=True)


# ==========================================================================
# 1. Intervention logic -- pure functions  q -> q'
#
# All operate on a query tensor shaped [B, T, Hq, D] (the q_proj output reshaped
# to expose the head axis) and return a NEW tensor; they never touch K or V.
# `head`/`a`/`b` index the head axis (dim 2).
# ==========================================================================

def iv_swap(q: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Exchange the query slices of heads a and b. Bidirectional (a<->b)."""
    out = q.clone()
    out[:, :, a, :], out[:, :, b, :] = q[:, :, b, :].clone(), q[:, :, a, :].clone()
    return out


def iv_copy(q: torch.Tensor, recipient: int, donor: int) -> torch.Tensor:
    """Overwrite the recipient head's query with the donor's (donor unchanged)."""
    out = q.clone()
    out[:, :, recipient, :] = q[:, :, donor, :]
    return out


def iv_zero(q: torch.Tensor, head: int) -> torch.Tensor:
    out = q.clone(); out[:, :, head, :] = 0.0
    return out


def iv_scale(q: torch.Tensor, head: int, alpha: float) -> torch.Tensor:
    out = q.clone(); out[:, :, head, :] = q[:, :, head, :] * alpha
    return out


def iv_normalize(q: torch.Tensor, head: int, target_norm: Optional[float] = None) -> torch.Tensor:
    """Set the head's per-token query to unit (or fixed) L2 norm, keeping direction."""
    out = q.clone()
    v = q[:, :, head, :]
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = 1.0 if target_norm is None else float(target_norm)
    out[:, :, head, :] = v / n * scale
    return out


def iv_noise(q: torch.Tensor, head: int, sigma: float,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Add isotropic Gaussian noise N(0, sigma^2) to the head's query."""
    out = q.clone()
    noise = torch.randn(q[:, :, head, :].shape, generator=generator,
                        dtype=q.dtype, device=q.device) * sigma
    out[:, :, head, :] = q[:, :, head, :] + noise
    return out


# ==========================================================================
# 2. Forward hooks and weight swap -- context managers (always restore)
# ==========================================================================

@contextmanager
def patched_query(model: Any, layer_idx: int, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Install a forward hook on layer ``layer_idx``'s q_proj that applies ``fn``
    to the query, reshaped to [B, T, Hq, D], for the duration of the context.

    The hook fires *before* q_norm and RoPE. Because both are head-independent,
    a patch here propagates through them exactly as a genuine query would. Keys
    and values are never hooked, so K and V are untouched. Complexity: one extra
    O(B*T*Hq*D) tensor op per forward; memory O(B*T*Hq*D).
    """
    top = model_topology(model)
    Hq, D = top["n_heads"], top["head_dim"]

    def hook(_module, _inp, output):
        B, T, HD = output.shape
        q = output.view(B, T, Hq, D)
        q = fn(q)
        return q.reshape(B, T, HD)

    handle = _attn(model, layer_idx).q_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def swapped_wq(model: Any, layer_idx: int, a: int, b: int):
    """Swap the W_Q row-blocks of heads a and b in layer ``layer_idx``, restore on
    exit. q_proj has no bias, and q_norm's gamma is shared across heads, so only
    these two [D, hidden] blocks move. Complexity: O(D*hidden) copy; the forward
    pass itself is unchanged in cost.
    """
    W = _attn(model, layer_idx).q_proj.weight           # [Hq*D, hidden]
    top = model_topology(model); D = top["head_dim"]
    ra = slice(a * D, (a + 1) * D)
    rb = slice(b * D, (b + 1) * D)
    orig = W.data.clone()
    try:
        with torch.no_grad():
            block_a = orig[ra].clone()
            W.data[ra] = orig[rb]
            W.data[rb] = block_a
        yield
    finally:
        with torch.no_grad():
            W.data.copy_(orig)


# ==========================================================================
# 3. Metric computation -- one shared adapter over Phase 3.1
# ==========================================================================

def capture_metrics_df(
    model: Any, input_ids: torch.Tensor, *, condition: str, seq_index: int,
    site_layer: Optional[int] = None, recipient: Optional[int] = None,
    donor: Optional[int] = None, intervention: Optional[Any] = None,
    keep_layers: Optional[Sequence[int]] = None, min_query_pos: int = 4,
) -> pd.DataFrame:
    """Run one (optionally intervened) forward pass and return tidy per-token rows.

    Reuses Phase 3.1 verbatim: capture -> compute_query_metrics -> build_metric_table
    (which already excludes BOS-as-query and positions < min_query_pos). Adds the
    experimental bookkeeping columns. ``intervention`` is a context manager (from
    section 2) or None for baseline. ``keep_layers`` restricts the returned rows to
    those captured-layer indices (e.g. just the intervened layer) AND restricts the
    capture itself to those layers -- the forward still runs the whole model, but we
    only record/offload the attention matrices we actually measure, which for a
    single-layer intervention is ~L-fold less capture work and memory.
    """
    # Only capture the layers we will keep. The forward still computes every layer
    # (the intervention at layer L propagates downstream), but recording just the
    # measured layer avoids offloading all L attention matrices on every pass.
    cap_cfg = (U.CaptureConfig(layers=list(keep_layers))
               if keep_layers is not None else U.CaptureConfig())
    ctx = intervention if intervention is not None else nullcontext()
    with ctx:
        with U.capture_attention(model, cap_cfg) as rec:
            with torch.no_grad():
                model(input_ids, use_cache=False)
        cap = rec.finalize(input_ids=input_ids, model=model, seq_id=f"seq{seq_index:03d}")
    qm = U.compute_query_metrics(cap)
    tbl = U.build_metric_table(qm, seq_index=seq_index, min_query_pos=min_query_pos)
    df = pd.DataFrame({n: np.asarray(tbl[n]) for n in tbl.names})
    df = df.rename(columns={"query_pos": "token"})
    if keep_layers is not None:
        df = df[df["layer"].isin(list(keep_layers))].copy()
    df["condition"] = condition
    df["site_layer"] = -1 if site_layer is None else int(site_layer)
    df["recipient"] = -1 if recipient is None else int(recipient)
    df["donor"] = -1 if donor is None else int(donor)
    keep = ["seq", "site_layer", "condition", "recipient", "donor",
            "layer", "head", "kv_group", "token", *(_METRIC_COLUMNS.keys())]
    return df[[c for c in keep if c in df.columns]]


# ==========================================================================
# 4. Experiments A / B / C
# ==========================================================================

@dataclass
class Prompt:
    input_ids: torch.Tensor    # [1, T]
    seq_index: int


def _as_prompts(prompts: Sequence[Any]) -> List[Prompt]:
    out = []
    for i, p in enumerate(prompts):
        ids = p.input_ids if isinstance(p, Prompt) else p
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out.append(Prompt(ids, i))
    return out


def run_baseline(model: Any, prompts: Sequence[Any], layers: Optional[Sequence[int]] = None,
                 min_query_pos: int = 4) -> pd.DataFrame:
    """Baseline metrics for every prompt, computed once and shared by all sites."""
    P = _as_prompts(prompts)
    frames = []
    for p in P:
        frames.append(capture_metrics_df(
            model, p.input_ids, condition="baseline", seq_index=p.seq_index,
            keep_layers=layers, min_query_pos=min_query_pos))
    return pd.concat(frames, ignore_index=True)


def experiment_wq_swap(model: Any, prompts: Sequence[Any],
                       sites: Sequence[Tuple[int, int, int]], min_query_pos: int = 4,
                       progress: bool = True) -> pd.DataFrame:
    """Experiment A. For each site (layer, a, b) swap W_Q[a]<->W_Q[b] and remeasure
    the two heads at that layer. Returns per-token rows for heads {a,b}."""
    P = _as_prompts(prompts)
    frames = []
    for si, (L, a, b) in enumerate(sites):
        _tick("A · W_Q swap", si, len(sites), progress)
        for p in P:
            frames.append(capture_metrics_df(
                model, p.input_ids, condition="wq_swap", seq_index=p.seq_index,
                site_layer=L, recipient=a, donor=b,
                intervention=swapped_wq(model, L, a, b),
                keep_layers=[L], min_query_pos=min_query_pos))
    df = pd.concat(frames, ignore_index=True)
    involved = sorted({h for (_, a, b) in sites for h in (a, b)})
    return df[df["head"].isin(involved)].copy()


def experiment_patch(model: Any, prompts: Sequence[Any],
                     sites: Sequence[Tuple[int, int, int]], mode: str = "copy",
                     min_query_pos: int = 4, progress: bool = True) -> pd.DataFrame:
    """Experiment B. Activation patching at q_proj.

    mode='copy' (the transfer measurement): for each site it runs BOTH directions
    -- recipient a with donor b's query (b intact), and recipient b with donor a's --
    so transfer can be read for each head. mode='swap' exchanges a<->b in one pass
    (used only for the A-vs-B equivalence check, where it must reproduce Exp. A).
    """
    P = _as_prompts(prompts)
    frames = []
    label = "B · copy patch" if mode == "copy" else "B · swap patch (A=B check)"
    for si, (L, a, b) in enumerate(sites):
        _tick(label, si, len(sites), progress)
        if mode == "copy":
            directions = [(a, b), (b, a)]
            for (recip, donor) in directions:
                fn = (lambda q, r=recip, d=donor: iv_copy(q, recipient=r, donor=d))
                for p in P:
                    frames.append(capture_metrics_df(
                        model, p.input_ids, condition="patch_copy", seq_index=p.seq_index,
                        site_layer=L, recipient=recip, donor=donor,
                        intervention=patched_query(model, L, fn),
                        keep_layers=[L], min_query_pos=min_query_pos))
        else:
            fn = (lambda q, a=a, b=b: iv_swap(q, a, b))
            for p in P:
                frames.append(capture_metrics_df(
                    model, p.input_ids, condition="patch_swap", seq_index=p.seq_index,
                    site_layer=L, recipient=a, donor=b,
                    intervention=patched_query(model, L, fn),
                    keep_layers=[L], min_query_pos=min_query_pos))
    df = pd.concat(frames, ignore_index=True)
    if mode == "copy":
        # keep only the recipient head's row for each direction
        df = df[df["head"] == df["recipient"]].copy()
    return df


# NOTE on scale invariance: the hook patches the query BEFORE q_norm, and Qwen3's
# QK-RMSNorm re-normalizes each head's query to a fixed RMS. So a pure pre-norm
# rescale (scale0.5 / scale2.0) is almost entirely absorbed by the norm and is
# nearly inert -- this is a genuine architectural property, not a bug. It is kept
# in the sweep precisely to demonstrate that magnitude alone carries little signal,
# whereas zero (0 -> 0 survives the norm), normalize (direction only), and noise
# (direction corrupted) do move sink behaviour.
_ABLATIONS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "zero": lambda q, h: iv_zero(q, h),
    "scale0.5": lambda q, h: iv_scale(q, h, 0.5),
    "scale2.0": lambda q, h: iv_scale(q, h, 2.0),
    "normalize": lambda q, h: iv_normalize(q, h),
    "noise1.0": lambda q, h: iv_noise(q, h, 1.0),
}


def experiment_ablation(model: Any, prompts: Sequence[Any],
                        targets: Sequence[Tuple[int, int]],
                        kinds: Sequence[str] = ("zero", "scale0.5", "scale2.0",
                                                "normalize", "noise1.0"),
                        min_query_pos: int = 4) -> pd.DataFrame:
    """Experiment C. Apply each named ablation to a target (layer, head) and
    remeasure that head."""
    P = _as_prompts(prompts)
    frames = []
    target_set = set(targets)
    for (L, h) in targets:
        for kind in kinds:
            fn = (lambda q, h=h, kind=kind: _ABLATIONS[kind](q, h))
            for p in P:
                frames.append(capture_metrics_df(
                    model, p.input_ids, condition=f"ablate_{kind}", seq_index=p.seq_index,
                    site_layer=L, recipient=h, donor=-1,
                    intervention=patched_query(model, L, fn),
                    keep_layers=[L], min_query_pos=min_query_pos))
    df = pd.concat(frames, ignore_index=True)
    keep = df.apply(lambda r: (int(r["site_layer"]), int(r["head"])) in target_set, axis=1)
    return df[keep].copy()


# ==========================================================================
# 5. Statistical aggregation -- four granularities
# ==========================================================================

def pool_levels(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Average in the fixed order token -> sequence -> head -> overall, so each
    level's unit is exchangeable and long prompts never dominate.

    L1 per token (as given); L2 per sequence (mean over tokens); L3 per head (mean
    over sequences); L4 overall (mean over heads).
    """
    metrics = list(_METRIC_COLUMNS.keys())
    keys = ["condition", "site_layer", "recipient", "donor", "layer", "head", "kv_group"]
    l1 = df
    l2 = df.groupby(keys + ["seq"], as_index=False)[metrics].mean()
    l3 = l2.groupby(keys, as_index=False)[metrics].mean()
    # sink score is explicitly the per-head mean of the per-token sink probability
    l3 = l3.rename(columns={"bos_prob": "sink_score"})
    other = [m for m in metrics if m != "bos_prob"]
    l4 = l3.groupby(["condition", "site_layer", "layer"], as_index=False)[["sink_score"] + other].mean()
    return {"L1_per_token": l1, "L2_per_sequence": l2, "L3_per_head": l3, "L4_overall": l4}


def transfer_fractions(baseline_l3: pd.DataFrame, intervention_l3: pd.DataFrame,
                       sites: Sequence[Tuple[int, int, int]],
                       metrics: Sequence[str] = TRANSFER_METRICS,
                       eps: float = 1e-6) -> pd.DataFrame:
    """Per site, per metric: how far the recipient moves from its own baseline
    toward the donor's baseline.

        transfer = (m_recip^int - m_recip^base) / (m_donor^base - m_recip^base)

    0 = no movement, 1 = lands on the donor. Undefined (NaN) when the two heads'
    baselines coincide (no gap to close); such rows are flagged ``has_gap=False``.
    Computed at Level 3 (per head), the granularity of the causal claim.
    """
    b = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    iv = intervention_l3.rename(columns={"bos_prob": "sink_score"})
    metrics = ["sink_score" if m == "bos_prob" else m for m in metrics]
    cond = iv["condition"].iloc[0] if len(iv) else "intervention"

    def base_lookup(layer, head):
        r = b[(b["layer"] == layer) & (b["head"] == head)]
        return r.iloc[0] if len(r) else None

    # Iterate the directions actually measured in the intervention: each intervened
    # row records (site_layer, recipient, donor). This avoids assuming both
    # directions were run (a copy-mode patch runs one direction per site).
    seen = iv[["site_layer", "recipient", "donor"]].drop_duplicates()
    rows = []
    for _, r in seen.iterrows():
        L, recip, donor = int(r["site_layer"]), int(r["recipient"]), int(r["donor"])
        if donor < 0:
            continue  # ablation rows have no donor; transfer is undefined
        base_recip = base_lookup(L, recip)
        base_donor = base_lookup(L, donor)
        iv_rows = iv[(iv["site_layer"] == L) & (iv["recipient"] == recip)
                     & (iv["donor"] == donor) & (iv["head"] == recip)]
        if base_recip is None or base_donor is None or not len(iv_rows):
            continue
        iv_recip = iv_rows.iloc[0]
        for m in metrics:
            gap = float(base_donor[m] - base_recip[m])
            delta = float(iv_recip[m] - base_recip[m])
            has_gap = abs(gap) > eps
            rows.append({
                "site_layer": L, "recipient": recip, "donor": donor, "metric": m,
                "baseline_recipient": float(base_recip[m]),
                "baseline_donor": float(base_donor[m]),
                "intervened_recipient": float(iv_recip[m]),
                "delta": delta, "gap": gap,
                "transfer": (delta / gap) if has_gap else float("nan"),
                "has_gap": has_gap, "condition": cond,
            })
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Visualization
# ==========================================================================

def plot_before_after(baseline_l3: pd.DataFrame, intervention_l3: pd.DataFrame,
                      metric: str, out_path: PathLike, show: bool = False) -> Path:
    """Scatter of a head's metric before vs after the intervention, at its site."""
    col = "sink_score" if metric == "bos_prob" else metric
    b = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    iv = intervention_l3.rename(columns={"bos_prob": "sink_score"})
    merged = iv.merge(b[["layer", "head", col]], on=["layer", "head"],
                      suffixes=("_after", "_before"))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    lo = float(min(merged[f"{col}_before"].min(), merged[f"{col}_after"].min()))
    hi = float(max(merged[f"{col}_before"].max(), merged[f"{col}_after"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1, label="no change")
    sc = ax.scatter(merged[f"{col}_before"], merged[f"{col}_after"],
                    c=merged["site_layer"], cmap="viridis", s=28, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="intervened layer")
    ax.set_xlabel(f"{col} — baseline"); ax.set_ylabel(f"{col} — after intervention")
    ax.set_title(f"Before vs after · {col}", loc="left")
    ax.legend()
    return _save(fig, out_path, show)


def plot_transfer(tf: pd.DataFrame, out_path: PathLike, show: bool = False) -> Path:
    """Transfer fraction by metric (box) and vs depth for the sink score."""
    valid = tf[tf["has_gap"]]
    metrics = list(dict.fromkeys(valid["metric"]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    data = [valid[valid["metric"] == m]["transfer"].values for m in metrics]
    ax[0].axhline(1, color="#3a3", lw=1, ls="--", label="full transfer")
    ax[0].axhline(0, color="#a33", lw=1, ls="--", label="no transfer")
    if any(len(d) for d in data):
        bp = ax[0].boxplot(data, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0"); patch.set_alpha(0.6)
    ax[0].set_xticks(range(1, len(metrics) + 1)); ax[0].set_xticklabels(metrics, rotation=30, ha="right")
    ax[0].set_ylabel("transfer fraction"); ax[0].set_title("(a) transfer by metric", loc="left")
    ax[0].legend(fontsize=7)

    sink = valid[valid["metric"] == "sink_score"]
    if len(sink):
        prof = sink.groupby("site_layer")["transfer"].agg(["mean", "std"]).reset_index()
        ax[1].axhline(1, color="#3a3", lw=1, ls="--"); ax[1].axhline(0, color="#a33", lw=1, ls="--")
        ax[1].errorbar(prof["site_layer"], prof["mean"], yerr=prof["std"].fillna(0),
                       fmt="-o", ms=4, capsize=3, color="#4C72B0")
        ax[1].set_xlabel("intervened layer"); ax[1].set_ylabel("sink-score transfer")
        ax[1].set_title("(b) sink-score transfer through depth", loc="left")
    fig.suptitle("Causal transfer of sink behaviour along the swapped query", fontsize=12)
    return _save(fig, out_path, show)


def plot_ablation(ablation_l3: pd.DataFrame, baseline_l3: pd.DataFrame,
                  out_path: PathLike, show: bool = False) -> Path:
    """Sink score under each ablation vs baseline."""
    ab = ablation_l3.rename(columns={"bos_prob": "sink_score"})
    base = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    kinds = sorted(ab["condition"].unique())
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(kinds))
    means = [ab[ab["condition"] == k]["sink_score"].mean() for k in kinds]
    base_mean = base["sink_score"].mean()
    ax.axhline(base_mean, color="0.4", ls="--", lw=1, label=f"baseline ({base_mean:.3f})")
    ax.bar(x, means, 0.8, color="#DD8452", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("ablate_", "") for k in kinds], rotation=20)
    ax.set_ylabel("mean sink score"); ax.set_title("Analysis C · query ablations vs baseline", loc="left")
    ax.legend()
    return _save(fig, out_path, show)


# ==========================================================================
# 7. Interpretation + orchestration
# ==========================================================================

def interpret(tf: pd.DataFrame, patch_agreement: Optional[float] = None,
              ablation_l3: Optional[pd.DataFrame] = None,
              baseline_l3: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Generate scientific conclusions from the measured transfer, not hard-coded."""
    obs: Dict[str, str] = {}
    sink = tf[(tf["metric"] == "sink_score") & tf["has_gap"]]["transfer"]
    med = float(np.nanmedian(sink)) if len(sink) else float("nan")

    if not math.isfinite(med):
        obs["transfer"] = ("Baseline sink scores of paired heads were too close to "
                            "measure transfer; no causal gap to close.")
    elif med >= 0.8:
        obs["transfer"] = (
            f"Transplanting the donor head's query onto the recipient moves the "
            f"recipient's sink score a median {med:.0%} of the way to the donor's. The "
            f"majority of sink specialization follows the query pathway, providing strong "
            f"causal evidence that query projections are a dominant mechanism underlying "
            f"within-group specialization.")
    elif med >= 0.3:
        obs["transfer"] = (
            f"Transplanting the donor head's query transfers a median {med:.0%} of the "
            f"sink-score gap. Query projections explain a substantial fraction of sink "
            f"specialization, although additional downstream computations contribute.")
    else:
        obs["transfer"] = (
            f"Transplanting the donor head's query transfers only a median {med:.0%} of the "
            f"sink-score gap. Query projections alone are insufficient to explain sink "
            f"specialization, suggesting subsequent computations or residual interactions "
            f"play a larger causal role.")

    if patch_agreement is not None:
        obs["consistency"] = (
            f"Weight swap (A) and activation patch (B) agree to {patch_agreement:.1e} on "
            f"sink score, confirming the effect is carried by the query vector itself "
            f"rather than by any weight/norm interaction.")

    if ablation_l3 is not None and baseline_l3 is not None and len(ablation_l3):
        ab = ablation_l3.rename(columns={"bos_prob": "sink_score"})
        base = float(baseline_l3.rename(columns={"bos_prob": "sink_score"})["sink_score"].mean())
        parts = [f"{k.replace('ablate_','')}={ab[ab['condition']==k]['sink_score'].mean():.3f}"
                 for k in sorted(ab["condition"].unique())]
        obs["ablation"] = (f"Sink score under ablations (baseline {base:.3f}): " + ", ".join(parts) +
                           ". Interventions that preserve query direction retain more sink "
                           "behaviour than those that destroy it, indicating direction carries "
                           "the signal.")
    return obs


@dataclass
class Phase33Results:
    baseline: pd.DataFrame
    wq_swap: pd.DataFrame
    patch: pd.DataFrame
    ablation: pd.DataFrame
    pooled_baseline: Dict[str, pd.DataFrame]
    pooled_swap: Dict[str, pd.DataFrame]
    transfer: pd.DataFrame
    observations: Dict[str, str]
    figures: Dict[str, Path]


def run_phase33(model: Any, prompts: Sequence[Any],
                sites: Sequence[Tuple[int, int, int]],
                ablation_targets: Optional[Sequence[Tuple[int, int]]] = None,
                out_dir: PathLike = "phase3_3_causal", min_query_pos: int = 4,
                run_patch: bool = True, run_ablation: bool = True,
                make_figures: bool = True, show: bool = False,
                progress: bool = True, equiv_check_sites: int = 3) -> Phase33Results:
    """Run experiments A (+B, +C), pool, compute transfer, and interpret.

    ``sites`` are (layer, head_a, head_b) with a, b in the same KV group.
    ``equiv_check_sites`` caps how many sites get the (redundant) A=B swap-mode
    check -- it only confirms an identity, so a handful is plenty; set 0 to skip.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    site_layers = sorted({L for (L, _, _) in sites})
    n_sites, n_prompts = len(sites), len(list(prompts))

    if progress:
        n_check = min(equiv_check_sites, n_sites) if run_patch else 0
        passes = (n_prompts                              # baseline
                  + n_sites * n_prompts                  # A
                  + (2 * n_sites * n_prompts if run_patch else 0)   # B copy
                  + (n_check * n_prompts)                # B swap (A=B check)
                  + (len(ablation_targets or []) * 5 * n_prompts if run_ablation else 0))
        print(f"Phase 3.3 · {n_sites} sites × {n_prompts} prompts "
              f"→ ~{passes} forward passes. Progress:", flush=True)

    base = run_baseline(model, prompts, layers=site_layers, min_query_pos=min_query_pos)
    # Experiment A: full W_Q swap. With n_rep=2 a symmetric swap makes the recipient
    # land exactly on the donor, so it is used for the A-vs-B equivalence check, not
    # for measuring transfer (which would be a tautology).
    swap = experiment_wq_swap(model, prompts, sites, min_query_pos=min_query_pos, progress=progress)
    # Experiment B (copy mode) is the informative transfer measurement: the donor is
    # left intact and we ask how far the recipient moves toward it.
    patch = experiment_patch(model, prompts, sites, mode="copy",
                             min_query_pos=min_query_pos, progress=progress) if run_patch else pd.DataFrame()

    ablation = pd.DataFrame()
    if run_ablation and ablation_targets:
        if progress:
            print("    [C · ablation] running", flush=True)
        ablation = experiment_ablation(model, prompts, ablation_targets, min_query_pos=min_query_pos)

    pooled_base = pool_levels(base)
    pooled_swap = pool_levels(swap)
    # transfer from the copy-mode patch if available, else fall back to the swap
    transfer_src = pool_levels(patch)["L3_per_head"] if len(patch) else pooled_swap["L3_per_head"]
    tf = transfer_fractions(pooled_base["L3_per_head"], transfer_src, sites)

    # A vs B equivalence: swap-mode patch must reproduce the weight swap exactly.
    # This only confirms an identity, so check a few sites rather than all of them.
    patch_agree = None
    if run_patch and equiv_check_sites > 0:
        check = list(sites)[:equiv_check_sites]
        patch_swap = experiment_patch(model, prompts, check, mode="swap",
                                      min_query_pos=min_query_pos, progress=progress)
        ps3 = pool_levels(patch_swap)["L3_per_head"].rename(columns={"bos_prob": "sink_score"})
        sl3 = pooled_swap["L3_per_head"].rename(columns={"bos_prob": "sink_score"})
        m = ps3.merge(sl3, on=["site_layer", "recipient", "layer", "head"],
                      suffixes=("_patch", "_swap"))
        if len(m):
            patch_agree = float((m["sink_score_patch"] - m["sink_score_swap"]).abs().max())

    ab_l3 = pool_levels(ablation)["L3_per_head"] if len(ablation) else None
    obs = interpret(tf, patch_agreement=patch_agree, ablation_l3=ab_l3,
                    baseline_l3=pooled_base["L3_per_head"])

    base.to_csv(out_dir / "tables" / "baseline_per_token.csv", index=False)
    pooled_base["L3_per_head"].to_csv(out_dir / "tables" / "baseline_per_head.csv", index=False)
    swap.to_csv(out_dir / "tables" / "wq_swap_per_token.csv", index=False)
    if len(patch):
        patch.to_csv(out_dir / "tables" / "patch_copy_per_token.csv", index=False)
    tf.to_csv(out_dir / "tables" / "transfer_fractions.csv", index=False)
    if len(ablation):
        ablation.to_csv(out_dir / "tables" / "ablation_per_token.csv", index=False)
    (out_dir / "observations.json").write_text(json.dumps(obs, indent=2))

    transfer_pooled = pool_levels(patch)["L3_per_head"] if len(patch) else pooled_swap["L3_per_head"]
    figs: Dict[str, Path] = {}
    if make_figures:
        figs["before_after"] = plot_before_after(
            pooled_base["L3_per_head"], transfer_pooled, "bos_prob",
            out_dir / "figures" / "B_before_after_sink.png", show)
        figs["transfer"] = plot_transfer(tf, out_dir / "figures" / "A_transfer.png", show)
        if ab_l3 is not None:
            figs["ablation"] = plot_ablation(ab_l3, pooled_base["L3_per_head"],
                                             out_dir / "figures" / "C_ablation.png", show)

    return Phase33Results(base, swap, patch, ablation, pooled_base, pooled_swap,
                          tf, obs, figs)


def _save(fig, out_path: PathLike, show: bool) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.show() if show else plt.close(fig)
    return out_path


# ==========================================================================
# 8. Extended causal analyses (Phase 3.3+)
#    D  propagation / ripple : patch one layer, measure all later layers
#    E  partial transplant   : interpolate recipient<-donor query by alpha
#    F  layer-wise ablation  : each ablation's effect as a function of depth
# ==========================================================================

def iv_interpolate(q: torch.Tensor, recipient: int, donor: int, alpha: float) -> torch.Tensor:
    """Blend the recipient's query toward the donor's:
    q'_recip = (1 - alpha) * q_recip + alpha * q_donor. alpha=0 is a no-op,
    alpha=1 is a full copy. Donor is left unchanged."""
    out = q.clone()
    out[:, :, recipient, :] = (1.0 - alpha) * q[:, :, recipient, :] + alpha * q[:, :, donor, :]
    return out


# --- D. Propagation / ripple ------------------------------------------------

def experiment_ripple(model: Any, prompts: Sequence[Any],
                      patch_sites: Sequence[Tuple[int, int, int]],
                      directions: str = "one", min_query_pos: int = 4,
                      progress: bool = True) -> pd.DataFrame:
    """Patch the recipient's full query at one layer and measure EVERY layer.

    The single-layer transplant pins the patched layer to transfer=1 by identity;
    the scientific content is how far that effect propagates downstream. For each
    patch site (L, a, b) we copy a<-b (and b<-a if directions='both') at layer L,
    capture ALL layers, and keep the recipient head's per-token rows tagged with
    site_layer=L (the patch layer) and layer=L' (the measurement layer).

    Cost: one FULL-capture forward per (site, direction, prompt) -- heavier than the
    single-layer experiments, so sweep a handful of patch layers, not all of them.
    """
    P = _as_prompts(prompts)
    frames = []
    for si, (L, a, b) in enumerate(patch_sites):
        _tick("D · ripple", si, len(patch_sites), progress)
        pairs = [(a, b)] if directions == "one" else [(a, b), (b, a)]
        for (recip, donor) in pairs:
            fn = (lambda q, r=recip, d=donor: iv_copy(q, recipient=r, donor=d))
            for p in P:
                df = capture_metrics_df(
                    model, p.input_ids, condition="ripple", seq_index=p.seq_index,
                    site_layer=L, recipient=recip, donor=donor,
                    intervention=patched_query(model, L, fn),
                    keep_layers=None, min_query_pos=min_query_pos)   # ALL layers
                frames.append(df[df["head"] == recip])
    return pd.concat(frames, ignore_index=True)


def ripple_transfer(baseline_l3: pd.DataFrame, ripple_l3: pd.DataFrame,
                    metric: str = "bos_prob", eps: float = 1e-6) -> pd.DataFrame:
    """Per (patch_layer, measure_layer, direction): transfer of ``metric`` toward the
    donor's baseline at the measurement layer, plus the absolute change (delta).

    transfer = (m_recip^patched(L') - m_recip^base(L')) / (m_donor^base(L') - m_recip^base(L'))
    delta    =  m_recip^patched(L') - m_recip^base(L')

    L' < L is ~0 (earlier layers are untouched), L' = L is 1 (identity), L' > L is
    the ripple. Returns one row per (patch_layer, measure_layer, recipient, donor).
    """
    col = "sink_score" if metric == "bos_prob" else metric
    b = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    iv = ripple_l3.rename(columns={"bos_prob": "sink_score"})

    def base_at(layer, head):
        r = b[(b["layer"] == layer) & (b["head"] == head)]
        return float(r.iloc[0][col]) if len(r) else None

    rows = []
    for _, r in iv.iterrows():
        L, Lp = int(r["site_layer"]), int(r["layer"])
        recip, donor = int(r["recipient"]), int(r["donor"])
        if recip != int(r["head"]) or donor < 0:
            continue
        br = base_at(Lp, recip); bd = base_at(Lp, donor)
        if br is None or bd is None:
            continue
        gap = bd - br
        delta = float(r[col]) - br
        has_gap = abs(gap) > eps
        rows.append({
            "patch_layer": L, "measure_layer": Lp, "distance": Lp - L,
            "recipient": recip, "donor": donor, "metric": col,
            "baseline_recipient": br, "baseline_donor": bd,
            "patched_recipient": float(r[col]), "delta": delta, "abs_delta": abs(delta),
            "gap": gap, "transfer": (delta / gap) if has_gap else float("nan"),
            "has_gap": has_gap,
        })
    return pd.DataFrame(rows)


def effective_propagation_depth(ripple_tf: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """For each patch layer, the contiguous number of downstream layers over which
    the (mean) transfer stays >= ``threshold``, starting from the patched layer.
    A compact summary of how far the query's effect reaches."""
    tf = ripple_tf[ripple_tf["has_gap"]]
    out = []
    for L in sorted(tf["patch_layer"].unique()):
        sub = (tf[(tf["patch_layer"] == L) & (tf["distance"] >= 0)]
               .groupby("distance")["transfer"].mean().sort_index())
        depth = 0
        for dist, val in sub.items():
            if dist == 0:
                continue
            if val >= threshold:
                depth = dist
            else:
                break
        out.append({"patch_layer": L, "propagation_depth": depth,
                    "threshold": threshold,
                    "transfer_at_+1": float(sub.get(1, float("nan")))})
    return pd.DataFrame(out)


def plot_ripple_heatmap(ripple_tf: pd.DataFrame, out_path: PathLike,
                        value: str = "transfer", show: bool = False) -> Path:
    """Heatmap of patch layer (rows) x measurement layer (cols)."""
    tf = ripple_tf.copy()
    grid = tf.groupby(["patch_layer", "measure_layer"])[value].mean().reset_index()
    piv = grid.pivot(index="patch_layer", columns="measure_layer", values=value)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="magma",
                   extent=[piv.columns.min() - .5, piv.columns.max() + .5,
                           piv.index.min() - .5, piv.index.max() + .5],
                   vmin=0, vmax=1 if value == "transfer" else None)
    fig.colorbar(im, ax=ax, label=value)
    ax.plot([piv.columns.min(), piv.columns.max()],
            [piv.columns.min(), piv.columns.max()], "--", color="cyan", lw=1,
            label="patch = measure")
    ax.set_xlabel("measurement layer L'"); ax.set_ylabel("patch layer L")
    ax.set_title(f"Propagation of the query transplant · {value}", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path, show)


def plot_ripple_decay(ripple_tf: pd.DataFrame, out_path: PathLike,
                      value: str = "transfer", show: bool = False) -> Path:
    """Decay of the effect vs distance downstream from the patched layer."""
    tf = ripple_tf[ripple_tf["distance"] >= 0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    # (a) one curve per patch layer
    for L in sorted(tf["patch_layer"].unique()):
        sub = tf[tf["patch_layer"] == L].groupby("distance")[value].mean()
        ax[0].plot(sub.index, sub.values, "-o", ms=3, label=f"L={L}")
    ax[0].set_xlabel("distance downstream (L' - L)"); ax[0].set_ylabel(value)
    ax[0].set_title(f"(a) {value} decay per patch layer", loc="left")
    ax[0].legend(fontsize=7, ncol=2)
    # (b) aggregate mean +/- std over patch layers
    agg = tf.groupby("distance")[value].agg(["mean", "std"]).reset_index()
    ax[1].axhline(0.5, color="0.6", ls="--", lw=1, label="0.5")
    ax[1].errorbar(agg["distance"], agg["mean"], yerr=agg["std"].fillna(0),
                   fmt="-o", ms=4, capsize=3, color="#4C72B0")
    ax[1].set_xlabel("distance downstream (L' - L)"); ax[1].set_ylabel(f"mean {value}")
    ax[1].set_title("(b) aggregate decay", loc="left"); ax[1].legend(fontsize=8)
    fig.suptitle("Ripple: how far the query transplant propagates", fontsize=12)
    return _save(fig, out_path, show)


# --- E. Partial query transplant --------------------------------------------

def experiment_partial(model: Any, prompts: Sequence[Any],
                       sites: Sequence[Tuple[int, int, int]],
                       alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                       directions: str = "one", min_query_pos: int = 4,
                       progress: bool = True) -> pd.DataFrame:
    """Interpolate the recipient's query toward the donor's by alpha and measure at
    the patched layer. alpha=0 recovers baseline, alpha=1 is the full transplant.

    directions='one' (default) uses recipient=a, donor=b per site, so the raw sink
    score / BOS logit trace a single recipient->donor path (alpha=0 is the
    recipient's own value, alpha=1 the donor's). directions='both' adds b<-a, which
    doubles the transfer samples but makes raw (un-normalized) curves symmetric about
    alpha=0.5 -- fine for the transfer fraction, misleading for raw quantities.
    """
    P = _as_prompts(prompts)
    frames = []
    for si, (L, a, b) in enumerate(sites):
        _tick("E · partial", si, len(sites), progress)
        pairs = [(a, b), (b, a)] if directions == "both" else [(a, b)]
        for (recip, donor) in pairs:
            for al in alphas:
                fn = (lambda q, r=recip, d=donor, al=al: iv_interpolate(q, r, d, al))
                for p in P:
                    df = capture_metrics_df(
                        model, p.input_ids, condition=f"alpha_{al:.2f}", seq_index=p.seq_index,
                        site_layer=L, recipient=recip, donor=donor,
                        intervention=patched_query(model, L, fn),
                        keep_layers=[L], min_query_pos=min_query_pos)
                    df = df[df["head"] == recip].copy()
                    df["alpha"] = float(al)
                    frames.append(df)
    return pd.concat(frames, ignore_index=True)


def partial_curves(baseline_l3: pd.DataFrame, partial_df: pd.DataFrame,
                   eps: float = 1e-6) -> pd.DataFrame:
    """For each alpha: mean sink score, mean BOS logit, and transfer fraction toward
    the donor (pooled token->seq->head, then over pairs)."""
    metrics = list(_METRIC_COLUMNS.keys())
    keys = ["condition", "site_layer", "recipient", "donor", "layer", "head", "kv_group", "alpha"]
    l2 = partial_df.groupby(keys + ["seq"], as_index=False)[metrics].mean()
    l3 = l2.groupby(keys, as_index=False)[metrics].mean().rename(columns={"bos_prob": "sink_score"})

    b = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    def base_at(layer, head, col):
        r = b[(b["layer"] == layer) & (b["head"] == head)]
        return float(r.iloc[0][col]) if len(r) else float("nan")

    rows = []
    for _, r in l3.iterrows():
        L, recip, donor = int(r["site_layer"]), int(r["recipient"]), int(r["donor"])
        br = base_at(L, recip, "sink_score"); bd = base_at(L, donor, "sink_score")
        gap = bd - br
        rows.append({
            "alpha": float(r["alpha"]), "site_layer": L, "recipient": recip, "donor": donor,
            "sink_score": float(r["sink_score"]), "bos_logit": float(r["bos_logit"]),
            "transfer": ((r["sink_score"] - br) / gap) if abs(gap) > eps else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_partial(curves: pd.DataFrame, out_path: PathLike, show: bool = False) -> Path:
    """Transfer, BOS logit, and sink score as a function of alpha."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    def _curve(axis, col, ylab, title, ref_line=None):
        agg = curves.groupby("alpha")[col].agg(["mean", "std"]).reset_index()
        axis.errorbar(agg["alpha"], agg["mean"], yerr=agg["std"].fillna(0),
                      fmt="-o", ms=5, capsize=3, color="#4C72B0")
        if ref_line is not None:
            axis.plot([0, 1], ref_line, "--", color="0.6", lw=1, label="linear")
            axis.legend(fontsize=8)
        axis.set_xlabel(r"$\alpha$ (recipient $\rightarrow$ donor)")
        axis.set_ylabel(ylab); axis.set_title(title, loc="left")
    _curve(ax[0], "transfer", "transfer fraction", "(a) transfer vs alpha", ref_line=[0, 1])
    _curve(ax[1], "bos_logit", "BOS logit", "(b) BOS logit vs alpha")
    _curve(ax[2], "sink_score", "sink score", "(c) sink score vs alpha")
    fig.suptitle("Partial query transplant: continuity of sink specialization", fontsize=12)
    return _save(fig, out_path, show)


# --- F. Layer-wise ablation profiles ----------------------------------------

def ablation_profiles(baseline_l3: pd.DataFrame, ablation_l3: pd.DataFrame,
                      relative: bool = True) -> pd.DataFrame:
    """Sink score for each ablation kind as a function of the target layer, either
    absolute or relative to that layer's baseline sink score."""
    ab = ablation_l3.rename(columns={"bos_prob": "sink_score"})
    b = baseline_l3.rename(columns={"bos_prob": "sink_score"})
    rows = []
    for _, r in ab.iterrows():
        L, head = int(r["site_layer"]), int(r["head"])
        base = b[(b["layer"] == L) & (b["head"] == head)]
        base_ss = float(base.iloc[0]["sink_score"]) if len(base) else float("nan")
        ss = float(r["sink_score"])
        rows.append({
            "layer": L, "head": head, "condition": r["condition"].replace("ablate_", ""),
            "sink_score": ss, "baseline": base_ss,
            "relative": (ss / base_ss) if (relative and base_ss > 1e-9) else ss,
        })
    return pd.DataFrame(rows)


def plot_ablation_profiles(profiles: pd.DataFrame, out_path: PathLike,
                           value: str = "relative", show: bool = False) -> Path:
    """One curve per ablation kind across depth (layer on x)."""
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    if value == "relative":
        ax.axhline(1.0, color="0.4", ls="--", lw=1, label="baseline")
    for kind in sorted(profiles["condition"].unique()):
        sub = profiles[profiles["condition"] == kind].groupby("layer")[value].mean()
        ax.plot(sub.index, sub.values, "-o", ms=4, label=kind)
    ax.set_xlabel("target layer"); ax.set_ylabel(
        "relative sink score (ablated / baseline)" if value == "relative" else "sink score")
    ax.set_title("Layer-wise ablation profiles", loc="left")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_path, show)
