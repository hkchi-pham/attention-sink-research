"""
phase3_utils.py -- shared helpers for Phase 3 of the attention-sink project.

Single-file, flattened build of the validated ``attn_sink`` package so it can be
dropped into the Drive project folder the way Phase 1 used ``sink_lib.py`` and
Phase 2 used ``phase2_utils.py``. Logic is byte-identical to the package; only
the module boundaries are gone.

Phase 3.1 -- captures the attention computation BEFORE the softmax, derives
per-query sink statistics, validates them against analytic identities, and
persists everything so Phase 3.2 (shared-KV specialisation) and Phase 3.3
(query-pathway intervention) never re-run inference.

Sections
--------
  1. GQA plumbing + CaptureConfig / CaptureResult / AttentionRecorder
  2. Attention-interface registration (capture_attention)
  3. QueryMetrics + MetricTable
  4. Persistence
  5. Validation checks C1-C6
  6. Figures
  7. Extended analyses (variance / competitor / LSE / group ICC)
  8. Frozen-benchmark loader + deterministic prompt construction

Validated: 12/12 tests, 9/9 invariant checks, GQA ratios n_rep in {1, 2, 4},
against transformers 4.57.6 / torch 2.5.1.
"""

from __future__ import annotations

__version__ = "3.1.0"







# ==========================================================================
# SECTION: capture
# ==========================================================================
#
# Attention capture for Qwen3 (HF transformers).
#
# Mechanism
# ---------
# ``Qwen3Attention.forward`` computes q/k/v, applies q_norm/k_norm and RoPE, then
# dispatches to an *attention interface function*:
#
#     attention_interface = eager_attention_forward
#     if config._attn_implementation != "eager":
#         attention_interface = ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
#
# We register our own function under a private name and point the config at it.
# The function is a byte-for-byte replica of ``eager_attention_forward`` with
# recording inserted between the matmul and the softmax. That is the only place
# in the graph where pre-softmax logits exist, so it cannot be reached with an
# ``nn.Module`` forward hook.
#
# What the interface function receives
# ------------------------------------
#     query : (B, n_heads,    T, head_dim)   post-q_norm, post-RoPE
#     key   : (B, n_kv_heads, T, head_dim)   post-k_norm, post-RoPE, pre-repeat_kv
#     value : (B, n_kv_heads, T, head_dim)
#     scaling : float  == head_dim ** -0.5
#
# RoPE is a rotation, so ||q||, ||k|| here are identical to their post-QK-norm
# values. Qwen3 applies RMSNorm over head_dim only, which bounds these norms:
#
#     ||k||^2 = sum_i gamma_i^2 u_i^2,   ||u||^2 = head_dim
#
# so ||k|| in [sqrt(d)*min|gamma|, sqrt(d)*max|gamma|]. Recorded norms should be
# checked against that interval (see validate.check_qk_norm_bounds).
#
# Stored logits are RAW and UNMASKED (pure QK^T * scaling). The causal mask is
# applied downstream in metrics.py. This keeps the stored tensor a clean
# mathematical object and makes the masking convention explicit and auditable
# rather than baked in as -inf / finfo.min sentinels.
import contextlib
import datetime as _dt
import platform
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# GQA plumbing (re-implemented rather than imported, so the semantics are
# pinned here and cannot drift with a transformers upgrade)
# --------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Exact replica of transformers' ``repeat_kv``.

    (B, n_kv, T, D) -> (B, n_kv * n_rep, T, D)

    The expand-then-reshape ordering fixes the head mapping: output head index
    ``h`` is served by kv head ``h // n_rep``.
    """
    batch, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, head_dim)
    return hs.reshape(batch, n_kv * n_rep, slen, head_dim)


def kv_group_of_head(head_idx: int, n_rep: int) -> int:
    """Which KV head serves query head ``head_idx``. See ``repeat_kv``."""
    return head_idx // n_rep


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """What to record and where to put it.

    Memory: the [L, H, T, T] tensors dominate. In float32 they cost
    ``L*H*T*T*4`` bytes each -- for Qwen3-1.7B (L=28, H=16) that is 117 MB at
    T=256, 469 MB at T=512, 1.9 GB at T=1024. Set ``store_probs=False`` or
    restrict ``layers`` for long sequences; the derived [L, H, T] metrics are
    ~4 orders of magnitude smaller and are what 3.2 / 3.3 actually need.
    """

    # which of the six required quantities to keep
    store_logits: bool = True          # pre-softmax QK^T / sqrt(d), unmasked
    store_probs: bool = True           # post-softmax attention probabilities
    store_queries: bool = True         # post-RoPE query vectors
    store_keys: bool = True            # post-RoPE key vectors (compact, n_kv heads)
    store_norms: bool = True           # ||q||, ||k|| (cheap; kept even if vectors dropped)

    # subsetting
    layers: Optional[Sequence[int]] = None   # None = all layers

    # numerics / placement
    store_dtype: torch.dtype = torch.float32
    store_device: str = "cpu"

    # the position treated as the sink. 0 unless you know what you are doing.
    sink_position: int = 0

    def resolved_layers(self, n_layers: int) -> List[int]:
        if self.layers is None:
            return list(range(n_layers))
        bad = [l for l in self.layers if not (0 <= l < n_layers)]
        if bad:
            raise ValueError(f"layer indices out of range for {n_layers}-layer model: {bad}")
        return sorted(set(int(l) for l in self.layers))


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class CaptureResult:
    """One forward pass over one sequence, fully described.

    Tensor shapes (L = number of *captured* layers, H = query heads,
    G = kv heads, T = sequence length, D = head_dim):

        logits      [L, H, T, T]   raw, unmasked, = q . k / sqrt(D)
        probs       [L, H, T, T]   softmax over causally-valid keys
        queries     [L, H, T, D]   post-RoPE
        keys        [L, G, T, D]   post-RoPE, one entry per KV head
        query_norms [L, H, T]
        key_norms   [L, G, T]

    ``layer_index[i]`` gives the true model layer of captured slice ``i``.
    """

    logits: Optional[torch.Tensor]
    probs: Optional[torch.Tensor]
    queries: Optional[torch.Tensor]
    keys: Optional[torch.Tensor]
    query_norms: Optional[torch.Tensor]
    key_norms: Optional[torch.Tensor]

    layer_index: torch.Tensor          # [L] int64, true layer ids
    kv_group_of_head: torch.Tensor     # [H] int64, head -> kv head
    input_ids: torch.Tensor            # [T] int64
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience -----------------------------------------------------
    @property
    def n_layers(self) -> int:
        return int(self.layer_index.numel())

    @property
    def n_heads(self) -> int:
        return int(self.kv_group_of_head.numel())

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.numel())

    @property
    def n_rep(self) -> int:
        return int(self.meta["num_key_value_groups"])

    def causal_mask(self) -> torch.Tensor:
        """[T, T] bool; True where key s is visible to query t (s <= t)."""
        T = self.seq_len
        return torch.ones(T, T, dtype=torch.bool).tril_()

    def describe(self) -> str:
        lines = [
            f"CaptureResult  seq_id={self.meta.get('seq_id')!r}",
            f"  model            : {self.meta.get('model_name')}",
            f"  layers captured  : {self.n_layers} of {self.meta.get('num_hidden_layers')}",
            f"  heads / kv heads : {self.n_heads} / {self.meta.get('num_key_value_heads')}"
            f"  (n_rep={self.meta.get('num_key_value_groups')})",
            f"  seq_len          : {self.seq_len}",
            f"  sink position    : {self.meta.get('sink_position')} "
            f"(token id {self.meta.get('sink_token_id')}, "
            f"is_bos={self.meta.get('sink_is_bos')})",
            f"  store dtype      : {self.meta.get('store_dtype')}",
        ]
        for name in ("logits", "probs", "queries", "keys", "query_norms", "key_norms"):
            t = getattr(self, name)
            lines.append(f"  {name:<16} : {'--' if t is None else tuple(t.shape)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------

class AttentionRecorder:
    """Accumulates per-layer captures during a forward pass."""

    def __init__(self, cfg: CaptureConfig, n_layers: int):
        self.cfg = cfg
        self.n_layers = n_layers
        self._wanted = set(cfg.resolved_layers(n_layers))
        self._buf: Dict[int, Dict[str, torch.Tensor]] = {}
        self._module_meta: Dict[str, Any] = {}
        self.mask_was_none = False
        self.n_calls = 0

    # -- called from inside the attention interface function ---------------
    def record(
        self,
        layer_idx: int,
        raw_logits: torch.Tensor,   # (B, H, T, T) unmasked
        probs: torch.Tensor,        # (B, H, T, T)
        query: torch.Tensor,        # (B, H, T, D)
        key: torch.Tensor,          # (B, G, T, D)
        module: Any,
    ) -> None:
        self.n_calls += 1
        if layer_idx not in self._wanted:
            return
        if raw_logits.shape[0] != 1:
            raise NotImplementedError(
                "capture is restricted to batch size 1. Padding + left/right "
                "alignment would silently corrupt the causal-mask bookkeeping, "
                "and correctness is the priority here. Loop over prompts instead."
            )
        if layer_idx in self._buf:
            raise RuntimeError(
                f"layer {layer_idx} recorded twice in one capture. Are you "
                "generating with a KV cache? Phase 3.1 expects a single "
                "full-sequence forward pass (use_cache=False)."
            )

        dt, dev = self.cfg.store_dtype, self.cfg.store_device
        ent: Dict[str, torch.Tensor] = {}
        if self.cfg.store_logits:
            ent["logits"] = raw_logits[0].detach().to(device=dev, dtype=dt)
        if self.cfg.store_probs:
            ent["probs"] = probs[0].detach().to(device=dev, dtype=dt)
        if self.cfg.store_queries:
            ent["queries"] = query[0].detach().to(device=dev, dtype=dt)
        if self.cfg.store_keys:
            ent["keys"] = key[0].detach().to(device=dev, dtype=dt)
        if self.cfg.store_norms:
            # computed in float32 regardless of model dtype
            ent["query_norms"] = query[0].detach().float().norm(dim=-1).to(device=dev, dtype=dt)
            ent["key_norms"] = key[0].detach().float().norm(dim=-1).to(device=dev, dtype=dt)
        self._buf[layer_idx] = ent

        if not self._module_meta:
            self._module_meta = {
                "num_key_value_groups": int(getattr(module, "num_key_value_groups")),
                "head_dim": int(getattr(module, "head_dim")),
                "scaling": float(getattr(module, "scaling")),
            }

    # -- assembly ----------------------------------------------------------
    def finalize(
        self,
        input_ids: torch.Tensor,
        model: Any,
        seq_id: str = "seq0",
        tokenizer: Any = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> CaptureResult:
        if not self._buf:
            raise RuntimeError(
                "nothing was recorded. The custom attention implementation was "
                "probably not active -- check that capture_attention() wrapped "
                "the forward call and that the model is a Qwen3 model."
            )
        got = sorted(self._buf)
        want = self.cfg.resolved_layers(self.n_layers)
        if got != want:
            raise RuntimeError(f"captured layers {got} != requested {want}")

        def stack(name: str) -> Optional[torch.Tensor]:
            if name not in self._buf[got[0]]:
                return None
            return torch.stack([self._buf[l][name] for l in got], dim=0)

        ids = input_ids.detach().reshape(-1).cpu().to(torch.int64)
        cfgm = model.config
        n_rep = self._module_meta["num_key_value_groups"]
        n_heads = int(cfgm.num_attention_heads)

        sink_pos = self.cfg.sink_position
        sink_id = int(ids[sink_pos])
        bos_id = getattr(cfgm, "bos_token_id", None)

        meta: Dict[str, Any] = {
            "schema": "attn_sink.CaptureResult",
            "seq_id": seq_id,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "model_name": getattr(cfgm, "_name_or_path", None) or type(model).__name__,
            "model_type": getattr(cfgm, "model_type", None),
            "num_hidden_layers": int(cfgm.num_hidden_layers),
            "num_attention_heads": n_heads,
            "num_key_value_heads": int(cfgm.num_key_value_heads),
            "num_key_value_groups": n_rep,
            "head_dim": self._module_meta["head_dim"],
            "scaling": self._module_meta["scaling"],
            "rope_theta": float(getattr(cfgm, "rope_theta", float("nan"))),
            "rms_norm_eps": float(getattr(cfgm, "rms_norm_eps", float("nan"))),
            "model_dtype": str(next(model.parameters()).dtype),
            "store_dtype": str(self.cfg.store_dtype),
            "seq_len": int(ids.numel()),
            "sink_position": sink_pos,
            "sink_token_id": sink_id,
            "sink_is_bos": (bos_id is not None and sink_id == int(bos_id)),
            "bos_token_id": None if bos_id is None else int(bos_id),
            "attention_mask_was_none": self.mask_was_none,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "capture_config": {
                k: (str(v) if isinstance(v, torch.dtype) else v)
                for k, v in asdict(self.cfg).items()
            },
        }
        try:
            import transformers as _tf
            meta["transformers_version"] = _tf.__version__
        except Exception:  # pragma: no cover
            meta["transformers_version"] = "unknown"

        if tokenizer is not None:
            try:
                meta["tokens"] = tokenizer.convert_ids_to_tokens(ids.tolist())
                meta["text"] = tokenizer.decode(ids.tolist())
            except Exception as exc:  # pragma: no cover
                meta["tokens_error"] = repr(exc)
        if extra_meta:
            meta.update(extra_meta)

        if not meta["sink_is_bos"]:
            warnings.warn(
                f"position {sink_pos} holds token id {sink_id}, which is not the "
                f"model's bos_token_id ({bos_id}). Qwen3's tokenizer does not add "
                "a BOS token by default -- the 'BOS' quantities below are really "
                "'first-position' quantities. Use prepend_bos=True if you want a "
                "true BOS token.",
                stacklevel=2,
            )

        return CaptureResult(
            logits=stack("logits"),
            probs=stack("probs"),
            queries=stack("queries"),
            keys=stack("keys"),
            query_norms=stack("query_norms"),
            key_norms=stack("key_norms"),
            layer_index=torch.tensor(got, dtype=torch.int64),
            kv_group_of_head=torch.tensor(
                [kv_group_of_head(h, n_rep) for h in range(n_heads)], dtype=torch.int64
            ),
            input_ids=ids,
            meta=meta,
        )


# --------------------------------------------------------------------------
# Interface registration
# --------------------------------------------------------------------------

def _make_attention_fn(recorder: AttentionRecorder):
    """Build the recording attention function.

    Replica of ``transformers`` eager_attention_forward. Any deviation here is
    a scientific bug, so keep the arithmetic literally identical and only add
    the ``recorder.record`` call.
    """

    def sink_capture_attention_forward(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        raw_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        attn_weights = raw_logits
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            # HF returned no mask (some paths delegate causality to the kernel).
            # Apply it ourselves -- silently attending bidirectionally would
            # invalidate every number downstream.
            recorder.mask_was_none = True
            T_q, T_k = raw_logits.shape[-2], raw_logits.shape[-1]
            neg = torch.finfo(raw_logits.dtype).min
            tri = torch.ones(T_q, T_k, dtype=torch.bool, device=raw_logits.device).tril_()
            attn_weights = raw_logits.masked_fill(~tri, neg)

        probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        recorder.record(
            layer_idx=int(module.layer_idx),
            raw_logits=raw_logits,
            probs=probs,
            query=query,
            key=key,
            module=module,
        )

        probs_drop = F.dropout(probs, p=dropout, training=module.training)
        attn_output = torch.matmul(probs_drop, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, probs

    return sink_capture_attention_forward


def _set_attn_impl(model, name: str) -> Dict[int, str]:
    """Point the model (and any nested text config) at ``name``. Returns old values."""
    old: Dict[int, str] = {}
    seen = set()
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        old[id(cfg)] = getattr(cfg, "_attn_implementation", "eager")
        cfg._attn_implementation = name
    # sub-module configs sometimes hold their own copy
    for mod in model.modules():
        sub = getattr(mod, "config", None)
        if sub is not None and id(sub) not in seen:
            seen.add(id(sub))
            old[id(sub)] = getattr(sub, "_attn_implementation", "eager")
            sub._attn_implementation = name
    return old


def _restore_attn_impl(model, old: Dict[int, str]) -> None:
    seen = set()
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        if id(cfg) in old:
            cfg._attn_implementation = old[id(cfg)]
    for mod in model.modules():
        sub = getattr(mod, "config", None)
        if sub is not None and id(sub) not in seen:
            seen.add(id(sub))
            if id(sub) in old:
                sub._attn_implementation = old[id(sub)]


@contextlib.contextmanager
def capture_attention(model, cfg: Optional[CaptureConfig] = None):
    """Context manager that swaps in the recording attention implementation.

    Registers a uniquely-named attention function (and the matching *mask*
    function, so transformers still builds a real float causal mask), points
    the model's config at it, and restores the previous implementation on exit
    -- including on exception.

    Yields the :class:`AttentionRecorder`.
    """
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS

    cfg = cfg or CaptureConfig()
    n_layers = int(model.config.num_hidden_layers)
    recorder = AttentionRecorder(cfg, n_layers)

    name = f"_attn_sink_capture_{uuid.uuid4().hex[:12]}"
    AttentionInterface.register(name, _make_attention_fn(recorder))
    # reuse eager's mask builder so we receive a materialised additive mask
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    old = _set_attn_impl(model, name)
    try:
        yield recorder
    finally:
        _restore_attn_impl(model, old)
        for registry in (AttentionInterface, AttentionMaskInterface):
            try:
                registry._global_mapping.pop(name, None)
            except Exception:  # pragma: no cover - registry internals vary
                pass


# --------------------------------------------------------------------------
# Storage planning
# --------------------------------------------------------------------------

def estimate_capture_bytes(
    n_layers: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int,
    cfg: Optional[CaptureConfig] = None, itemsize: int = 4,
) -> Dict[str, int]:
    """Bytes a single capture will occupy, per tensor. Call before you run.

    The [L, H, T, T] matrices scale quadratically in T and dominate everything
    else by two orders of magnitude at realistic lengths.
    """
    cfg = cfg or CaptureConfig()
    L = len(cfg.resolved_layers(n_layers))
    T, D, H, G = seq_len, head_dim, n_heads, n_kv_heads
    out: Dict[str, int] = {}
    if cfg.store_logits:
        out["logits"] = L * H * T * T * itemsize
    if cfg.store_probs:
        out["probs"] = L * H * T * T * itemsize
    if cfg.store_queries:
        out["queries"] = L * H * T * D * itemsize
    if cfg.store_keys:
        out["keys"] = L * G * T * D * itemsize
    if cfg.store_norms:
        out["query_norms"] = L * H * T * itemsize
        out["key_norms"] = L * G * T * itemsize
    out["metrics_L_H_T"] = L * H * T * 8 * 10       # float64 QueryMetrics fields
    out["total"] = sum(v for k, v in out.items() if k != "total")
    return out


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"



# ==========================================================================
# SECTION: metrics
# ==========================================================================
#
# Per-query-token sink statistics derived from a CaptureResult.
#
# All quantities are in *nats*, computed in float64, over the causally-valid key
# set only. For query position t the valid keys are s in [0, t]; the "competitor"
# set excludes the sink position, so it is s in [0, t] \\ {sink}.
#
# The central identity
# --------------------
# With softmax taken over exactly the causally-valid keys,
#
#     p_sink(t) = exp(l_sink) / sum_{s<=t} exp(l_s)
#               = 1 / (1 + exp(LSE_comp(t) - l_sink(t)))
#               = sigmoid( margin_lse(t) )
#
# This is exact, not approximate. Two consequences worth internalising:
#
# 1.  It is the strongest available end-to-end check on the pipeline
#     (validate.py asserts it), and
#
# 2.  the requested "log-sum-exp margin vs BOS probability" plot is therefore
#     *analytically determined* -- it must trace the logistic curve and nothing
#     else. It is a correctness figure, not a discovery figure. The scientific
#     content lives in the other three: how much of the variation in p_sink is
#     carried by l_sink alone versus by the competitor term.
#
# Saturation
# ----------
# dp/dl = p(1-p), so above p ~ 0.95 the probability scale compresses differences
# of several nats into a few thousandths. Phase 3.2 should compare heads on
# ``margin_lse`` (unsaturated, additive) rather than on ``bos_prob``.
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


_NEG_INF = float("-inf")


@dataclass
class QueryMetrics:
    """Per (layer, head, query position). All tensors are [L, H, T] float64.

    ``valid`` is [T] bool: False at the sink position itself (no competitors
    exist there), True elsewhere. Every consumer must honour it.
    """

    bos_prob: torch.Tensor          # p(attend to sink position)
    bos_logit: torch.Tensor         # l_sink = q_t . k_sink / sqrt(D)
    comp_max_logit: torch.Tensor    # max over competing keys
    comp_lse_logit: torch.Tensor    # logsumexp over competing keys
    margin_max: torch.Tensor        # bos_logit - comp_max_logit
    margin_lse: torch.Tensor        # bos_logit - comp_lse_logit
    entropy: torch.Tensor           # -sum p log p over valid keys, nats
    entropy_norm: torch.Tensor      # entropy / log(n_valid_keys)
    argmax_key: torch.Tensor        # [L,H,T] int64, argmax over valid keys
    query_norm: Optional[torch.Tensor]   # [L,H,T]
    key_norm: Optional[torch.Tensor]     # [L,G,T] (per KV head, not per head)

    n_valid_keys: torch.Tensor      # [T] int64, = t + 1
    valid: torch.Tensor             # [T] bool
    layer_index: torch.Tensor       # [L] int64
    kv_group_of_head: torch.Tensor  # [H] int64

    # --- competitor structure (needs full logits; computed at capture time) ---
    # Optional so that QueryMetrics saved before these existed still load.
    comp_top2_logit: Optional[torch.Tensor] = None   # 2nd-largest competitor logit
    comp_top3_logit: Optional[torch.Tensor] = None   # 3rd-largest competitor logit
    comp_entropy: Optional[torch.Tensor] = None       # H of competitor-only softmax, nats
    comp_neff: Optional[torch.Tensor] = None          # exp(comp_entropy) = effective #competitors

    meta: Dict[str, Any] = field(default_factory=dict)

    def key_norm_per_head(self) -> Optional[torch.Tensor]:
        """Broadcast the per-KV-head key norms out to per-query-head [L,H,T]."""
        if self.key_norm is None:
            return None
        return self.key_norm.index_select(1, self.kv_group_of_head)


def compute_query_metrics(
    cap: CaptureResult,
    sink_position: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float64,
) -> QueryMetrics:
    """Derive the sink statistics from a capture.

    Requires ``cap.logits`` and ``cap.probs``.
    """
    if cap.logits is None or cap.probs is None:
        raise ValueError(
            "compute_query_metrics needs both logits and probs; re-capture with "
            "store_logits=True and store_probs=True."
        )

    sink = cap.meta["sink_position"] if sink_position is None else sink_position
    L, H, T, T2 = cap.logits.shape
    if T != T2:
        raise ValueError(f"logits must be square in the last two dims, got {(T, T2)}")
    if not (0 <= sink < T):
        raise ValueError(f"sink_position {sink} out of range for T={T}")

    logits = cap.logits.to(compute_dtype)
    probs = cap.probs.to(compute_dtype)

    dev = logits.device
    pos = torch.arange(T, device=dev)
    causal = pos.unsqueeze(1) >= pos.unsqueeze(0)          # [T,T] True where s <= t
    comp = causal.clone()
    comp[:, sink] = False                                  # drop the sink column

    # ---- BOS / sink column ------------------------------------------------
    # .clone() is load-bearing, not defensive: a bare slice is a strided VIEW of
    # the [L,H,T,T] matrix, and torch.save serialises a view's entire underlying
    # storage. Without it each saved QueryMetrics silently drags the full
    # logits+probs matrices along -- ~264 MB per prompt at L=28,H=16,T=192,
    # which defeats the point of dropping them from the capture.
    bos_logit = logits[..., sink].clone()                  # [L,H,T]
    bos_prob = probs[..., sink].clone()

    # ---- competitors ------------------------------------------------------
    masked = logits.masked_fill(~comp, _NEG_INF)           # [L,H,T,T]
    comp_max_logit, _ = masked.max(dim=-1)
    comp_lse_logit = torch.logsumexp(masked, dim=-1)       # raw here, NaN-filled below

    valid = comp.any(dim=-1)                               # [T] bool
    n_comp = comp.sum(dim=-1)                              # [T] number of competitors

    # ---- competitor structure: top-k, competitor entropy, N_eff -----------
    # Competitor-only softmax p_i = exp(l_i - LSE_comp) over competitor keys.
    # These need the full logit row, so they are computed HERE (logits in memory)
    # and saved into the lean QueryMetrics -- Phases 3.2/3.3 then reuse them
    # without the [L,H,T,T] matrices and without re-running inference.
    lp = masked - comp_lse_logit.unsqueeze(-1)            # [L,H,T,T]; -inf at masked
    p_comp = lp.exp()                                     # 0 at masked entries
    ent_term = torch.where(comp & torch.isfinite(lp), p_comp * lp, torch.zeros_like(p_comp))
    comp_entropy = -ent_term.sum(dim=-1)                  # [L,H,T]
    comp_neff = comp_entropy.exp()

    k = min(3, masked.shape[-1])
    topv = torch.topk(masked, k=k, dim=-1).values         # [L,H,T,k], -inf where absent

    def _kth(j: int) -> torch.Tensor:
        if j >= k:
            return torch.full_like(comp_max_logit, torch.nan)
        need = (n_comp >= (j + 1)).view(1, 1, T)          # enough competitors to define it
        return torch.where(need, topv[..., j], torch.full_like(comp_max_logit, torch.nan))

    comp_top2_logit = _kth(1)
    comp_top3_logit = _kth(2)

    # query position == sink has an empty competitor set -> mark everything invalid
    def _mask_invalid(x: torch.Tensor) -> torch.Tensor:
        return torch.where(valid.view(1, 1, T), x, torch.full_like(x, torch.nan))

    comp_max_logit  = _mask_invalid(comp_max_logit)
    comp_lse_logit  = _mask_invalid(comp_lse_logit)
    comp_entropy    = _mask_invalid(comp_entropy)
    comp_neff       = _mask_invalid(comp_neff)
    comp_top2_logit = _mask_invalid(comp_top2_logit)
    comp_top3_logit = _mask_invalid(comp_top3_logit)

    margin_max = bos_logit - comp_max_logit
    margin_lse = bos_logit - comp_lse_logit

    # ---- entropy over causally valid keys ---------------------------------
    p = probs.masked_fill(~causal, 0.0)
    plogp = torch.where(p > 0, p * p.clamp_min(1e-300).log(), torch.zeros_like(p))
    entropy = -plogp.sum(dim=-1)                           # [L,H,T]

    n_valid_keys = causal.sum(dim=-1).to(torch.int64)      # [T] = t+1
    denom = n_valid_keys.to(compute_dtype).log()
    entropy_norm = torch.where(
        denom > 0, entropy / denom.clamp_min(1e-300), torch.full_like(entropy, torch.nan)
    )

    argmax_key = probs.masked_fill(~causal, _NEG_INF).argmax(dim=-1)

    qn = cap.query_norms.to(compute_dtype) if cap.query_norms is not None else None
    kn = cap.key_norms.to(compute_dtype) if cap.key_norms is not None else None

    meta = dict(cap.meta)
    meta.update({
        "schema": "attn_sink.QueryMetrics",
        "sink_position": sink,
        "units": "nats",
        "compute_dtype": str(compute_dtype),
    })

    return QueryMetrics(
        bos_prob=bos_prob,
        bos_logit=bos_logit,
        comp_max_logit=comp_max_logit,
        comp_lse_logit=comp_lse_logit,
        margin_max=margin_max,
        margin_lse=margin_lse,
        entropy=entropy,
        entropy_norm=entropy_norm,
        argmax_key=argmax_key,
        query_norm=qn,
        key_norm=kn,
        n_valid_keys=n_valid_keys,
        valid=valid,
        layer_index=cap.layer_index.clone(),
        kv_group_of_head=cap.kv_group_of_head.clone(),
        comp_top2_logit=comp_top2_logit,
        comp_top3_logit=comp_top3_logit,
        comp_entropy=comp_entropy,
        comp_neff=comp_neff,
        meta=meta,
    )


# --------------------------------------------------------------------------
# Long-format table -- what the plots and Phases 3.2 / 3.3 consume
# --------------------------------------------------------------------------

_FLOAT_COLS = (
    "bos_prob", "bos_logit", "comp_max_logit", "comp_lse_logit",
    "margin_max", "margin_lse", "entropy", "entropy_norm",
    "comp_top2_logit", "comp_top3_logit", "comp_entropy", "comp_neff",
    "query_norm", "key_norm",
)
_INT_COLS = ("seq", "layer", "head", "kv_group", "query_pos", "n_valid_keys", "argmax_key")


@dataclass
class MetricTable:
    """Flat, tidy table. Every column is a 1-D tensor of identical length.

    One row per (sequence, layer, head, query position), sink position and any
    other invalid rows already dropped. This is the hand-off format: Phase 3.2
    groups by ``kv_group``, Phase 3.3 joins interventions on
    ``(seq, layer, head, query_pos)``.
    """

    columns: Dict[str, torch.Tensor]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(next(iter(self.columns.values())).numel())

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.columns[key]

    @property
    def names(self) -> List[str]:
        return list(self.columns)

    def select(self, mask: torch.Tensor) -> "MetricTable":
        return MetricTable({k: v[mask] for k, v in self.columns.items()}, dict(self.meta))

    def where(self, **eq) -> "MetricTable":
        """``tbl.where(layer=12, kv_group=3)``."""
        m = torch.ones(len(self), dtype=torch.bool)
        for k, v in eq.items():
            m &= self.columns[k] == v
        return self.select(m)

    def to_numpy(self) -> Dict[str, Any]:
        return {k: v.cpu().numpy() for k, v in self.columns.items()}

    @staticmethod
    def concat(tables: Sequence["MetricTable"]) -> "MetricTable":
        if not tables:
            raise ValueError("nothing to concatenate")
        names = tables[0].names
        for t in tables[1:]:
            if t.names != names:
                raise ValueError("column mismatch between tables")
        cols = {n: torch.cat([t[n] for t in tables], dim=0) for n in names}
        meta = {"schema": "attn_sink.MetricTable", "n_sources": len(tables),
                "sources": [t.meta.get("seq_id") for t in tables]}
        return MetricTable(cols, meta)

    def summary(self) -> str:
        lines = [f"MetricTable  rows={len(self)}  cols={len(self.columns)}"]
        for n in self.names:
            v = self.columns[n]
            if v.is_floating_point():
                fin = v[torch.isfinite(v)]
                if fin.numel() == 0:
                    lines.append(f"  {n:<16} all non-finite")
                else:
                    lines.append(
                        f"  {n:<16} mean={fin.mean():+.4f}  sd={fin.std():.4f}  "
                        f"min={fin.min():+.4f}  max={fin.max():+.4f}"
                        + (f"  [{v.numel()-fin.numel()} non-finite]"
                           if fin.numel() != v.numel() else "")
                    )
            else:
                lines.append(f"  {n:<16} {int(v.min())}..{int(v.max())}")
        return "\n".join(lines)


def build_metric_table(
    qm: QueryMetrics,
    seq_index: int = 0,
    drop_invalid: bool = True,
    min_query_pos: int = 1,
) -> MetricTable:
    """Flatten [L, H, T] metrics into tidy rows.

    ``min_query_pos`` drops the earliest query positions, where the competitor
    set is tiny (at t=1 there is exactly one competitor, so margin_max ==
    margin_lse identically and every statistic is degenerate). Default 1 only
    removes the sink row; consider 4-8 for plots.
    """
    L, H, T = qm.bos_prob.shape
    li = qm.layer_index.reshape(L, 1, 1).expand(L, H, T)
    hi = torch.arange(H).reshape(1, H, 1).expand(L, H, T)
    gi = qm.kv_group_of_head.reshape(1, H, 1).expand(L, H, T)
    ti = torch.arange(T).reshape(1, 1, T).expand(L, H, T)
    si = torch.full((L, H, T), int(seq_index), dtype=torch.int64)
    nk = qm.n_valid_keys.reshape(1, 1, T).expand(L, H, T)

    knph = qm.key_norm_per_head()
    cols: Dict[str, torch.Tensor] = {
        "seq": si.reshape(-1),
        "layer": li.reshape(-1),
        "head": hi.reshape(-1),
        "kv_group": gi.reshape(-1),
        "query_pos": ti.reshape(-1),
        "n_valid_keys": nk.reshape(-1),
        "argmax_key": qm.argmax_key.reshape(-1),
        "bos_prob": qm.bos_prob.reshape(-1),
        "bos_logit": qm.bos_logit.reshape(-1),
        "comp_max_logit": qm.comp_max_logit.reshape(-1),
        "comp_lse_logit": qm.comp_lse_logit.reshape(-1),
        "margin_max": qm.margin_max.reshape(-1),
        "margin_lse": qm.margin_lse.reshape(-1),
        "entropy": qm.entropy.reshape(-1),
        "entropy_norm": qm.entropy_norm.reshape(-1),
        "comp_top2_logit": (qm.comp_top2_logit.reshape(-1) if qm.comp_top2_logit is not None
                            else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
        "comp_top3_logit": (qm.comp_top3_logit.reshape(-1) if qm.comp_top3_logit is not None
                            else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
        "comp_entropy": (qm.comp_entropy.reshape(-1) if qm.comp_entropy is not None
                         else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
        "comp_neff": (qm.comp_neff.reshape(-1) if qm.comp_neff is not None
                      else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
        "query_norm": (qm.query_norm.reshape(-1) if qm.query_norm is not None
                       else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
        "key_norm": (knph.reshape(-1) if knph is not None
                     else torch.full((L * H * T,), torch.nan, dtype=torch.float64)),
    }
    tbl = MetricTable(cols, {
        "schema": "attn_sink.MetricTable",
        "seq_id": qm.meta.get("seq_id"),
        "sink_position": qm.meta.get("sink_position"),
        "units": "nats",
        "model_name": qm.meta.get("model_name"),
        "seq_len": T,
    })

    if drop_invalid:
        keep = qm.valid.reshape(1, 1, T).expand(L, H, T).reshape(-1).clone()
        keep &= cols["query_pos"] >= int(min_query_pos)
        keep &= torch.isfinite(cols["margin_lse"])
        tbl = tbl.select(keep)
    return tbl



# ==========================================================================
# SECTION: store
# ==========================================================================
#
# Persistence.
#
# Everything is a plain ``torch.save`` payload plus a human-readable JSON
# sidecar, so Phase 3.2 / 3.3 can load tensors without importing anything from
# this package except the dataclasses, and so a human can read the metadata
# without loading a gigabyte of attention matrices.
#
# Layout produced by run_phase31.py::
#
#     out/
#       captures/seq000.pt        CaptureResult payload
#       captures/seq000.json      metadata only
#       metrics/seq000_qm.pt      QueryMetrics payload
#       metrics/table.pt          pooled MetricTable
#       metrics/table.json        metadata + column names
#       figures/*.png
#       manifest.json
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch


SCHEMA_VERSION = "3.1.0"

PathLike = Union[str, os.PathLike]

_CAPTURE_TENSORS = (
    "logits", "probs", "queries", "keys", "query_norms", "key_norms",
    "layer_index", "kv_group_of_head", "input_ids",
)
_QM_TENSORS = (
    "bos_prob", "bos_logit", "comp_max_logit", "comp_lse_logit",
    "margin_max", "margin_lse", "entropy", "entropy_norm", "argmax_key",
    "comp_top2_logit", "comp_top3_logit", "comp_entropy", "comp_neff",
    "query_norm", "key_norm", "n_valid_keys", "valid",
    "layer_index", "kv_group_of_head",
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _write_sidecar(path: Path, meta: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"schema_version": SCHEMA_VERSION, **_jsonable(meta)}
    if extra:
        payload.update(_jsonable(extra))
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


# --------------------------------------------------------------------------

def save_capture(cap: CaptureResult, path: PathLike, write_sidecar: bool = True,
                 drop: Sequence[str] = ()) -> Path:
    """Persist a capture.

    ``drop`` omits named tensors from the file (they are written as None). Use
    ``drop=("logits", "probs")`` when disk is scarce: the [L,H,T,T] matrices are
    ~99% of the payload, and Phases 3.2 / 3.3 need only q, k and the derived
    [L,H,T] metrics. Note that a capture saved this way can no longer be
    re-validated with C1/C3 or have metrics recomputed from it -- compute and
    save QueryMetrics *before* dropping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bad = [d for d in drop if d not in _CAPTURE_TENSORS]
    if bad:
        raise ValueError(f"unknown tensor name(s) in drop: {bad}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "CaptureResult",
        "tensors": {n: (None if n in drop else getattr(cap, n)) for n in _CAPTURE_TENSORS},
        "meta": {**cap.meta, "dropped_tensors": list(drop)},
    }
    torch.save(payload, path)
    if write_sidecar:
        shapes = {n: (None if t is None else list(t.shape))
                  for n, t in payload["tensors"].items()}
        _write_sidecar(path.with_suffix(".json"), payload["meta"],
                       {"tensor_shapes": shapes,
                        "file_bytes": path.stat().st_size})
    return path


def load_capture(path: PathLike, map_location: str = "cpu") -> CaptureResult:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("kind") != "CaptureResult":
        raise ValueError(f"{path} is not a CaptureResult payload (kind={payload.get('kind')!r})")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema mismatch: file {payload.get('schema_version')!r} vs "
            f"package {SCHEMA_VERSION!r}. Re-run capture or write a migration."
        )
    return CaptureResult(meta=payload["meta"], **payload["tensors"])


def save_query_metrics(qm: QueryMetrics, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "kind": "QueryMetrics",
        "tensors": {n: getattr(qm, n) for n in _QM_TENSORS},
        "meta": qm.meta,
    }, path)
    return path


def load_query_metrics(path: PathLike, map_location: str = "cpu") -> QueryMetrics:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("kind") != "QueryMetrics":
        raise ValueError(f"{path} is not a QueryMetrics payload")
    return QueryMetrics(meta=payload["meta"], **payload["tensors"])


def save_metric_table(tbl: MetricTable, path: PathLike, write_sidecar: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "kind": "MetricTable",
        "columns": tbl.columns,
        "meta": tbl.meta,
    }, path)
    if write_sidecar:
        _write_sidecar(path.with_suffix(".json"), tbl.meta,
                       {"n_rows": len(tbl), "columns": tbl.names})
    return path


def load_metric_table(path: PathLike, map_location: str = "cpu") -> MetricTable:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("kind") != "MetricTable":
        raise ValueError(f"{path} is not a MetricTable payload")
    return MetricTable(payload["columns"], payload["meta"])


def write_manifest(out_dir: PathLike, entries: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "manifest.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION, **_jsonable(entries)}, indent=2))
    return p



# ==========================================================================
# SECTION: validate
# ==========================================================================
#
# Self-consistency checks.
#
# These are the reason to trust anything downstream. Each check is an identity
# that must hold exactly (up to floating point) if the capture is correct; a
# failure localises the bug rather than just signalling one.
#
#     C1  softmax(causally-masked stored logits) == stored probs
#         -> the recorded tensor really is pre-softmax, and the masking
#            convention used in metrics.py matches the one the model used.
#
#     C2  recompute logits from stored q, k (with repeat_kv) == stored logits
#         -> q/k storage is intact and the GQA head->group mapping (h // n_rep)
#            is the one the model actually used. This is the check Phase 3.2
#            depends on.
#
#     C3  probs sum to 1 over causally valid keys, and are ~0 elsewhere
#         -> causality was enforced.
#
#     C4  p_sink == sigmoid(margin_lse)
#         -> end-to-end: capture, masking, logsumexp and the sink column all
#            agree. See metrics.py for the derivation.
#
#     C5  ||q||, ||k|| lie inside the interval forced by Qwen3's QK-RMSNorm,
#         [sqrt(D)*min|gamma|, sqrt(D)*max|gamma|]
#         -> confirms QK-norm is active and quantifies how much dynamic range
#            the magnitude channel actually has. Requires the live model.
#
#     C6  entropy is within [0, log(n_valid_keys)].
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch



@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    tol: Optional[float] = None


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str,
            value: Optional[float] = None, tol: Optional[float] = None) -> None:
        self.checks.append(Check(name, passed, detail, value, tol))

    def __str__(self) -> str:
        w = max((len(c.name) for c in self.checks), default=4)
        lines = []
        for c in self.checks:
            flag = "PASS" if c.passed else "FAIL"
            num = "" if c.value is None else f"  ({c.value:.3e} vs tol {c.tol:.1e})"
            lines.append(f"  [{flag}] {c.name:<{w}}  {c.detail}{num}")
        lines.append(f"  ==> {'ALL CHECKS PASSED' if self.ok else 'FAILURES PRESENT'}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            bad = [c.name for c in self.checks if not c.passed]
            raise AssertionError(f"validation failed: {bad}\n{self}")


def _tol_for(dtype: torch.dtype, kind: str) -> float:
    """Tolerances chosen for the *storage* dtype, not float64 compute dtype."""
    base = {torch.float32: 1e-5, torch.float64: 1e-11, torch.bfloat16: 5e-2,
            torch.float16: 5e-3}.get(dtype, 1e-4)
    return base * (10.0 if kind == "logit" else 1.0)


def validate_capture(
    cap: CaptureResult,
    qm: Optional[QueryMetrics] = None,
    model: Optional[Any] = None,
    verbose: bool = False,
) -> ValidationReport:
    rep = ValidationReport()
    T = cap.seq_len
    causal = cap.causal_mask()
    dt = cap.logits.dtype if cap.logits is not None else torch.float32

    # ---- C1 -------------------------------------------------------------
    if cap.logits is not None and cap.probs is not None:
        masked = cap.logits.double().masked_fill(~causal, float("-inf"))
        recon = torch.softmax(masked, dim=-1)
        err = (recon - cap.probs.double()).abs().max().item()
        tol = _tol_for(dt, "prob")
        rep.add("C1_softmax_roundtrip", err < tol,
                "softmax(masked stored logits) == stored probs", err, tol)

    # ---- C2 -------------------------------------------------------------
    if cap.queries is not None and cap.keys is not None and cap.logits is not None:
        n_rep = cap.n_rep
        scaling = float(cap.meta["scaling"])
        # keys is [L, G, T, D]; repeat_kv treats dim 0 as batch, so L slots in
        # directly and the result is [L, H, T, D].
        k_rep = repeat_kv(cap.keys.double(), n_rep)                   # [L,H,T,D]
        recomputed = torch.matmul(cap.queries.double(), k_rep.transpose(-1, -2)) * scaling
        err = (recomputed - cap.logits.double()).abs().max().item()
        scale = cap.logits.double().abs().max().item() or 1.0
        tol = _tol_for(dt, "logit") * max(1.0, scale)
        rep.add("C2_qk_recompute", err < tol,
                f"q @ repeat_kv(k).T * {scaling:.6g} == stored logits "
                f"(head->kv map h//{n_rep})", err, tol)

        # C2b: prove the mapping is not accidentally symmetric under a
        # different grouping, otherwise C2 would be vacuous.
        if n_rep > 1 and cap.n_heads > 1:
            n_kv = cap.keys.shape[1]
            wrong = cap.kv_group_of_head.roll(1)  # any other permutation
            k_wrong = cap.keys.double().index_select(1, wrong)
            rec_w = torch.matmul(cap.queries.double(), k_wrong.transpose(-1, -2)) * scaling
            err_w = (rec_w - cap.logits.double()).abs().max().item()
            rep.add("C2b_mapping_is_identifying", err_w > tol * 10,
                    "a shifted head->kv mapping does NOT reproduce the logits",
                    err_w, tol * 10)

    # ---- C3 -------------------------------------------------------------
    if cap.probs is not None:
        p = cap.probs.double()
        s = p.masked_fill(~causal, 0.0).sum(-1)
        err_sum = (s - 1.0).abs().max().item()
        leak = p.masked_fill(causal, 0.0).abs().max().item()
        tol = _tol_for(dt, "prob")
        rep.add("C3a_prob_normalised", err_sum < tol,
                "probs sum to 1 over causal keys", err_sum, tol)
        rep.add("C3b_causality", leak < tol,
                "probs are zero on masked (future) keys", leak, tol)

    # ---- C4 -------------------------------------------------------------
    if qm is not None:
        v = qm.valid
        pred = torch.sigmoid(qm.margin_lse[..., v])
        obs = qm.bos_prob[..., v]
        err = (pred - obs).abs().max().item()
        tol = _tol_for(dt, "prob") * 10
        rep.add("C4_sigmoid_identity", err < tol,
                "p_sink == sigmoid(bos_logit - logsumexp(competitors))", err, tol)

    # ---- C5 -------------------------------------------------------------
    if model is not None and cap.query_norms is not None:
        rep_c5 = _check_qk_norm_bounds(cap, model)
        rep.checks.extend(rep_c5.checks)

    # ---- C6 -------------------------------------------------------------
    if qm is not None:
        nk = qm.n_valid_keys.to(torch.float64).clamp_min(1.0).log()
        e = qm.entropy
        lo_ok = bool((e >= -1e-9).all())
        hi_ok = bool((e <= nk.reshape(1, 1, -1) + 1e-9).all())
        rep.add("C6_entropy_bounds", lo_ok and hi_ok,
                "0 <= H(t) <= log(n_valid_keys(t))")

    if verbose:
        print(rep)
    return rep


def _check_qk_norm_bounds(cap: CaptureResult, model: Any) -> ValidationReport:
    """Qwen3 applies RMSNorm over head_dim to q and k before RoPE, and RoPE is
    a rotation, so the stored (post-RoPE) norms must satisfy

        ||q|| in [sqrt(D) * min|gamma_Q|, sqrt(D) * max|gamma_Q|]

    exactly. Violating this means either QK-norm is absent (wrong model) or the
    wrong tensor was captured.
    """
    rep = ValidationReport()
    D = float(cap.meta["head_dim"])
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        rep.add("C5_qk_norm_bounds", True, "skipped: could not locate model.layers")
        return rep

    worst_q = worst_k = 0.0
    ratio_q = ratio_k = float("nan")
    for i, lidx in enumerate(cap.layer_index.tolist()):
        attn = layers[lidx].self_attn
        for gamma_name, norms, tag in (
            ("q_norm", cap.query_norms, "q"),
            ("k_norm", cap.key_norms, "k"),
        ):
            g = getattr(attn, gamma_name, None)
            if g is None or norms is None:
                continue
            gamma = g.weight.detach().double().abs()
            lo, hi = math.sqrt(D) * gamma.min().item(), math.sqrt(D) * gamma.max().item()
            n = norms[i].double()
            below = (lo - n).clamp_min(0).max().item()
            above = (n - hi).clamp_min(0).max().item()
            v = max(below, above)
            if tag == "q":
                worst_q = max(worst_q, v); ratio_q = hi / max(lo, 1e-12)
            else:
                worst_k = max(worst_k, v); ratio_k = hi / max(lo, 1e-12)

    tol = 1e-3
    rep.add("C5a_q_norm_bounds", worst_q < tol,
            f"||q|| inside QK-norm interval (max achievable spread x{ratio_q:.1f})",
            worst_q, tol)
    rep.add("C5b_k_norm_bounds", worst_k < tol,
            f"||k|| inside QK-norm interval (max achievable spread x{ratio_k:.1f})",
            worst_k, tol)
    return rep



# ==========================================================================
# SECTION: plots
# ==========================================================================
#
# Figures for Phase 3.1.
#
# The four requested comparisons, all against BOS attention probability:
#     (a) BOS logit
#     (b) strongest-competitor margin
#     (c) log-sum-exp margin
#     (d) attention entropy
#
# Read them in that order and panel (c) will look suspiciously perfect. It is:
# p_sink == sigmoid(margin_lse) is an algebraic identity, so (c) can only ever
# trace the logistic curve. It is plotted as a *validation* panel with the
# analytic curve overlaid -- any deviation is a bug, not a finding.
#
# That makes (a) the interesting one. The gap between (a)'s scatter and (c)'s
# curve is exactly the contribution of the competitor term: if (a) is already
# tight, the sink is driven by the BOS logit alone; if (a) is diffuse, the model
# is controlling the sink mainly by suppressing competitors.
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
# Only force a headless backend when we are NOT inside IPython/Colab -- otherwise
# we would suppress inline rendering in the notebook.
try:
    import IPython
    _IN_IPYTHON = IPython.get_ipython() is not None
except Exception:
    _IN_IPYTHON = False
if not _IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


PathLike = Union[str, Path]

_PANELS: List[Tuple[str, str, str]] = [
    ("bos_logit",   "BOS logit  $\\ell_{t,0}$  (nats)",                          "(a)"),
    ("margin_max",  "margin vs strongest competitor  $\\ell_{t,0}-\\max_{s>0}\\ell_{t,s}$", "(b)"),
    ("margin_lse",  "margin vs LSE of competitors  $\\ell_{t,0}-\\mathrm{LSE}_{s>0}\\ell_{t,s}$", "(c)"),
    ("entropy",     "attention entropy  $H_t$  (nats)",                          "(d)"),
]


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=cap, replace=False)


def _logit_forward(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def plot_sink_panels(
    tbl: MetricTable,
    out_path: PathLike,
    max_points: int = 40_000,
    seed: int = 0,
    prob_scale: str = "linear",
    point_size: float = 2.0,
    alpha: float = 0.25,
    title: Optional[str] = None,
    show: bool = False,
) -> Path:
    """The four requested scatter comparisons, coloured by layer depth.

    ``prob_scale='logit'`` re-expresses the y axis as log(p/(1-p)), which
    un-saturates the top of the range. Recommended whenever a large fraction of
    points sit above p = 0.9.
    """
    d = tbl.to_numpy()
    idx = _subsample(len(tbl), max_points, seed)
    layer = d["layer"][idx]
    p = d["bos_prob"][idx].astype(float)
    y = _logit_forward(p) if prob_scale == "logit" else p
    ylabel = ("BOS attention probability  $p_{t,0}$" if prob_scale == "linear"
              else "$\\mathrm{logit}\\,p_{t,0}$")

    norm = Normalize(vmin=float(d["layer"].min()), vmax=float(d["layer"].max()))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
    for ax, (col, xlabel, tag) in zip(axes.ravel(), _PANELS):
        x = d[col][idx].astype(float)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=point_size, alpha=alpha,
                   c=cmap(norm(layer[good])), linewidths=0, rasterized=True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{tag}  {col} vs BOS probability", fontsize=11, loc="left")
        ax.grid(alpha=0.2, linewidth=0.5)

        if col == "margin_lse":
            xs = np.linspace(np.nanpercentile(x[good], 0.1),
                             np.nanpercentile(x[good], 99.9), 400)
            ref = xs if prob_scale == "logit" else 1.0 / (1.0 + np.exp(-xs))
            ax.plot(xs, ref, color="crimson", lw=1.4, zorder=5,
                    label="analytic  $p=\\sigma(\\mathrm{margin})$")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        if col == "entropy":
            ax.axvline(0.0, color="0.5", lw=0.8, ls=":")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.015, location="right")
    cb.set_label("layer index", fontsize=10)

    n_shown = len(idx)
    head = title or (f"Phase 3.1 -- pre-softmax sink structure  "
                     f"({tbl.meta.get('model_name', 'model')})")
    fig.suptitle(f"{head}\n{n_shown:,} of {len(tbl):,} (layer, head, query) points"
                 f"   |   sink position {tbl.meta.get('sink_position')}",
                 fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.show() if show else plt.close(fig)
    return out_path


def plot_layer_head_maps(tbl: MetricTable, out_path: PathLike, show: bool = False) -> Path:
    """Diagnostic heatmaps: mean sink probability and mean LSE margin per
    (layer, head), plus the within-KV-group spread that Phase 3.2 formalises.
    """
    d = tbl.to_numpy()
    layers = np.unique(d["layer"])
    heads = np.unique(d["head"])
    li = {v: i for i, v in enumerate(layers)}
    hi = {v: i for i, v in enumerate(heads)}

    grids: Dict[str, np.ndarray] = {}
    for col in ("bos_prob", "margin_lse"):
        g = np.full((len(layers), len(heads)), np.nan)
        acc = np.zeros_like(g)
        cnt = np.zeros_like(g)
        np.add.at(acc, ([li[v] for v in d["layer"]], [hi[v] for v in d["head"]]),
                  np.nan_to_num(d[col], nan=0.0))
        np.add.at(cnt, ([li[v] for v in d["layer"]], [hi[v] for v in d["head"]]),
                  np.isfinite(d[col]).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            g = acc / cnt
        grids[col] = g

    # within-group spread of the mean LSE margin (the Phase 3.2 quantity)
    kv_of_head = np.zeros(len(heads), dtype=int)
    for h, gvals in zip(d["head"], d["kv_group"]):
        kv_of_head[hi[h]] = gvals
    n_groups = int(kv_of_head.max()) + 1
    spread = np.full((len(layers), n_groups), np.nan)
    for gi in range(n_groups):
        cols = np.where(kv_of_head == gi)[0]
        if cols.size > 1:
            spread[:, gi] = np.nanmax(grids["margin_lse"][:, cols], axis=1) - \
                            np.nanmin(grids["margin_lse"][:, cols], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    specs = [
        (grids["bos_prob"], "mean $p_{t,0}$", "magma", "head"),
        (grids["margin_lse"], "mean LSE margin (nats)", "coolwarm", "head"),
        (spread, "within-KV-group spread of\nmean LSE margin (nats)", "cividis", "KV group"),
    ]
    for ax, (g, name, cm, xlab) in zip(axes, specs):
        im = ax.imshow(g, aspect="auto", origin="lower", cmap=cm,
                       extent=[-0.5, g.shape[1] - 0.5, layers.min() - 0.5, layers.max() + 0.5])
        ax.set_xlabel(xlab); ax.set_ylabel("layer"); ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Phase 3.1 diagnostics -- per (layer, head) sink structure", fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.show() if show else plt.close(fig)
    return out_path


def plot_margin_decomposition(tbl: MetricTable, out_path: PathLike, show: bool = False) -> Path:
    """How the LSE margin splits into its two additive parts, against query
    position. ``logsumexp`` over competitors grows like log(t) when the
    competitor logits are roughly exchangeable; if p_sink is flat in t then
    the BOS logit must be tracking that growth.
    """
    d = tbl.to_numpy()
    t = d["query_pos"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for ax, (cols, name) in zip(axes, [
        ((("bos_logit", "BOS logit"), ("comp_lse_logit", "LSE of competitors")),
         "logit terms vs query position"),
        ((("margin_lse", "LSE margin"), ("margin_max", "max margin")),
         "margins vs query position"),
    ]):
        for col, lbl in cols:
            v = d[col].astype(float)
            ok = np.isfinite(v)
            ts = np.unique(t[ok])
            med = np.array([np.median(v[ok][t[ok] == u]) for u in ts])
            q1 = np.array([np.percentile(v[ok][t[ok] == u], 25) for u in ts])
            q3 = np.array([np.percentile(v[ok][t[ok] == u], 75) for u in ts])
            ln, = ax.plot(ts, med, lw=1.6, label=lbl)
            ax.fill_between(ts, q1, q3, alpha=0.18, color=ln.get_color())
        ax.set_xlabel("query position $t$"); ax.set_ylabel("nats")
        ax.set_title(name, fontsize=11); ax.grid(alpha=0.2, lw=0.5); ax.legend(fontsize=9)
        ax.set_xscale("log")

    axes[0].set_title("logit terms vs query position\n"
                      "(LSE grows ~log t if competitors are exchangeable)", fontsize=10)
    fig.suptitle("Phase 3.1 -- additive decomposition of the sink margin", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.show() if show else plt.close(fig)
    return out_path



# ==========================================================================
# SECTION: analysis
# ==========================================================================
#
# Extended pre-softmax analyses for Phase 3.1.
#
# All four are analysis-only: they consume the pooled MetricTable (and nothing
# else), never re-run the model, and write reusable tensors for Phase 3.2. Three
# of the four are pure functions of scalars already in the table; competitor
# structure relies on the top-k / competitor-entropy fields, which were computed
# at capture time (see metrics.py) precisely so this stays inference-free.
#
# Everything is computed in float64 and is NaN-aware: rows are filtered to finite
# values per grouping cell before any variance, correlation, or ICC.
#
# The four analyses
# -----------------
# 1. variance_decomposition   Var(margin) = Var(BOS) + Var(LSE) - 2Cov, per layer.
# 2. competitor_structure     top-1/2/3, competitor entropy, N_eff, vs LSE/margin.
# 3. lse_decomposition        LSE = l_max + R, R = -log p_max^comp (exact).
# 4. group_statistics         within/between-KV-group variance and ICC(1).
#
# Two exact identities anchor the interpretation and are asserted in the tests:
#
#   * Var(margin) - [Var(BOS)+Var(LSE)-2Cov]  ==  0   (variance decomposition)
#   * p_max^competitor  ==  exp(-R)  where R = LSE_comp - l_max   (LSE split)
#
# The second says the residual R is literally the negative log-share of the single
# strongest competitor: R small <=> one dominant competitor, R large <=> many.
# That is the exact question analysis (3) is posed to answer.
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
import numpy as np


PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# numeric helpers (all float64, NaN-aware, population moments)
# --------------------------------------------------------------------------

def _finite(*cols: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Row-wise drop of any position where any column is non-finite."""
    keep = torch.ones_like(cols[0], dtype=torch.bool)
    for c in cols:
        keep &= torch.isfinite(c)
    return tuple(c[keep] for c in cols)


def _var(x: torch.Tensor) -> float:
    """Population variance (unbiased=False), so the decomposition identity is exact."""
    if x.numel() < 1:
        return float("nan")
    return float(((x - x.mean()) ** 2).mean())


def _cov(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 1:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean())


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = _finite(x, y)
    if x.numel() < 2:
        return float("nan")
    sx, sy = x.std(unbiased=False), y.std(unbiased=False)
    if float(sx) == 0.0 or float(sy) == 0.0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _nanmean(x: torch.Tensor) -> float:
    x = x[torch.isfinite(x)]
    return float(x.mean()) if x.numel() else float("nan")


def _layers_of(table: MetricTable) -> List[int]:
    return sorted(int(v) for v in torch.unique(table["layer"]).tolist())


def _col(table: MetricTable, name: str) -> torch.Tensor:
    return table[name].to(torch.float64)


# ==========================================================================
# 1. Variance decomposition
# ==========================================================================

@dataclass
class VarianceDecomposition:
    layers: torch.Tensor          # [L] int64
    var_bos: torch.Tensor         # [L]
    var_lse: torch.Tensor
    cov: torch.Tensor
    var_margin: torch.Tensor
    recon: torch.Tensor           # var_bos + var_lse - 2 cov  (should == var_margin)
    resid: torch.Tensor           # var_margin - recon
    rho: torch.Tensor             # corr(bos, lse)
    contrib_bos: torch.Tensor     # Cov(margin, bos)  = var_bos - cov
    contrib_lse: torch.Tensor     # Cov(margin, -lse) = var_lse - cov  (sums to var_margin)
    by_group: Optional[Dict[str, torch.Tensor]] = None   # per (layer, kv_group) tensors
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_identity_error(self) -> float:
        r = self.resid[torch.isfinite(self.resid)]
        return float(r.abs().max()) if r.numel() else float("nan")

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        d = {k: getattr(self, k) for k in
             ("layers", "var_bos", "var_lse", "cov", "var_margin", "recon",
              "resid", "rho", "contrib_bos", "contrib_lse")}
        if self.by_group is not None:
            for k, v in self.by_group.items():
                d[f"by_group__{k}"] = v
        return d


def variance_decomposition(table: MetricTable, include_kv_group: bool = True) -> VarianceDecomposition:
    """Var(margin_lse) decomposed into BOS-logit and competitor-LSE variance.

    Since margin_lse == bos_logit - comp_lse_logit exactly, the identity
    Var(margin) = Var(BOS) + Var(LSE) - 2 Cov(BOS, LSE) holds to float64.
    Verified via ``resid`` (asserted ~0 in the notebook).

    Per layer, the variance is pooled over all (head, query position, sequence)
    rows in that layer -- so it measures the *total* margin variability at that
    depth, including between-head mean differences. The optional per-(layer,
    kv_group) breakdown removes cross-group mixing.
    """
    layers = _layers_of(table)
    lay = table["layer"]
    b_all, l_all, m_all = _col(table, "bos_logit"), _col(table, "comp_lse_logit"), _col(table, "margin_lse")

    rows = {k: [] for k in ("var_bos", "var_lse", "cov", "var_margin",
                            "recon", "resid", "rho", "contrib_bos", "contrib_lse")}
    for L in layers:
        sel = lay == L
        b, l, m = _finite(b_all[sel], l_all[sel], m_all[sel])
        vb, vl, vm = _var(b), _var(l), _var(m)
        cv = _cov(b, l)
        recon = vb + vl - 2 * cv
        rho = cv / math.sqrt(vb * vl) if (vb > 0 and vl > 0) else float("nan")
        rows["var_bos"].append(vb); rows["var_lse"].append(vl); rows["cov"].append(cv)
        rows["var_margin"].append(vm); rows["recon"].append(recon)
        rows["resid"].append(vm - recon); rows["rho"].append(rho)
        rows["contrib_bos"].append(vb - cv); rows["contrib_lse"].append(vl - cv)

    by_group = None
    if include_kv_group:
        groups = sorted(int(v) for v in torch.unique(table["kv_group"]).tolist())
        G = len(groups)
        grp = table["kv_group"]
        vb_g = torch.full((len(layers), G), torch.nan, dtype=torch.float64)
        cov_g = torch.full((len(layers), G), torch.nan, dtype=torch.float64)
        vm_g = torch.full((len(layers), G), torch.nan, dtype=torch.float64)
        for i, L in enumerate(layers):
            for j, gv in enumerate(groups):
                sel = (lay == L) & (grp == gv)
                b, l, m = _finite(b_all[sel], l_all[sel], m_all[sel])
                if b.numel() >= 2:
                    vb_g[i, j] = _var(b); cov_g[i, j] = _cov(b, l); vm_g[i, j] = _var(m)
        by_group = {"groups": torch.tensor(groups), "var_bos": vb_g, "cov": cov_g, "var_margin": vm_g}

    T = lambda x: torch.tensor(x, dtype=torch.float64)
    return VarianceDecomposition(
        layers=torch.tensor(layers, dtype=torch.int64),
        var_bos=T(rows["var_bos"]), var_lse=T(rows["var_lse"]), cov=T(rows["cov"]),
        var_margin=T(rows["var_margin"]), recon=T(rows["recon"]), resid=T(rows["resid"]),
        rho=T(rows["rho"]), contrib_bos=T(rows["contrib_bos"]), contrib_lse=T(rows["contrib_lse"]),
        by_group=by_group,
        meta={"schema": "attn_sink.VarianceDecomposition", "units": "nats^2",
              "note": "margin_lse = bos_logit - comp_lse_logit; identity Var(m)=Var(b)+Var(l)-2Cov"},
    )


def plot_variance_decomposition(vd: VarianceDecomposition, out_path: PathLike,
                                show: bool = False) -> Path:
    L = vd.layers.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    ax[0].plot(L, vd.var_bos, "-o", ms=3, label="Var(BOS logit)")
    ax[0].plot(L, vd.var_lse, "-o", ms=3, label="Var(competitor LSE)")
    ax[0].plot(L, 2 * vd.cov, "-o", ms=3, label="2·Cov(BOS, LSE)")
    ax[0].plot(L, vd.var_margin, "-k", lw=2, label="Var(margin)")
    ax[0].plot(L, vd.recon, "--", color="crimson", lw=1.3, label="Var(B)+Var(L)−2Cov")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("variance (nats²)")
    ax[0].set_title("(a) variance terms", loc="left"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    cb, cl = vd.contrib_bos.numpy(), vd.contrib_lse.numpy()
    ax[1].bar(L, cb, label="Cov(margin, BOS)", color="#4C72B0")
    ax[1].bar(L, cl, bottom=cb, label="Cov(margin, −LSE)", color="#DD8452")
    ax[1].plot(L, vd.var_margin, "k_", ms=8, label="Var(margin) (= sum)")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("contribution to Var(margin) (nats²)")
    ax[1].set_title("(b) stacked attribution", loc="left"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].plot(L, vd.rho, "-o", ms=3, color="#55A868")
    ax[2].set_ylim(-1.05, 1.05)
    ax[2].set_xlabel("layer"); ax[2].set_ylabel("ρ(BOS logit, competitor LSE)")
    ax[2].set_title("(c) co-modulation", loc="left"); ax[2].grid(alpha=0.25)

    fig.suptitle(f"Variance decomposition of the sink margin   "
                 f"(max identity error {vd.max_identity_error:.1e} nats²)", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# 2. Competitor structure
# ==========================================================================

@dataclass
class CompetitorStructure:
    layers: torch.Tensor
    mean_top1: torch.Tensor
    mean_top2: torch.Tensor
    mean_top3: torch.Tensor
    mean_comp_entropy: torch.Tensor
    mean_neff: torch.Tensor
    r_neff_margin: torch.Tensor       # per-layer corr(N_eff, margin_lse)
    r_neff_lse: torch.Tensor          # per-layer corr(N_eff, comp_lse)
    r_top1_margin: torch.Tensor       # per-layer corr(top-1 competitor, margin_lse)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in
                ("layers", "mean_top1", "mean_top2", "mean_top3", "mean_comp_entropy",
                 "mean_neff", "r_neff_margin", "r_neff_lse", "r_top1_margin")}


def competitor_structure(table: MetricTable) -> CompetitorStructure:
    """Top-1/2/3 competitor logits, competitor entropy and N_eff = e^H, and how
    N_eff co-varies with the LSE and the margin across layers.

    Requires the competitor-structure columns (comp_top2/3, comp_neff). If they
    are absent the caller is on an old capture and should re-run Phase 3.1.
    """
    for need in ("comp_top2_logit", "comp_neff", "comp_entropy"):
        col = table[need]
        if not torch.isfinite(col).any():
            raise ValueError(
                f"column '{need}' is all-NaN. This table predates the competitor-"
                "structure fields; re-run the Phase 3.1 capture with the current "
                "phase3_utils to populate them (no full matrices needed)."
            )
    layers = _layers_of(table)
    lay = table["layer"]
    t1, t2, t3 = _col(table, "comp_max_logit"), _col(table, "comp_top2_logit"), _col(table, "comp_top3_logit")
    ce, ne = _col(table, "comp_entropy"), _col(table, "comp_neff")
    lse, mar = _col(table, "comp_lse_logit"), _col(table, "margin_lse")

    def per_layer(fn):
        return torch.tensor([fn(lay == L) for L in layers], dtype=torch.float64)

    return CompetitorStructure(
        layers=torch.tensor(layers, dtype=torch.int64),
        mean_top1=per_layer(lambda s: _nanmean(t1[s])),
        mean_top2=per_layer(lambda s: _nanmean(t2[s])),
        mean_top3=per_layer(lambda s: _nanmean(t3[s])),
        mean_comp_entropy=per_layer(lambda s: _nanmean(ce[s])),
        mean_neff=per_layer(lambda s: _nanmean(ne[s])),
        r_neff_margin=per_layer(lambda s: _pearson(ne[s], mar[s])),
        r_neff_lse=per_layer(lambda s: _pearson(ne[s], lse[s])),
        r_top1_margin=per_layer(lambda s: _pearson(t1[s], mar[s])),
        meta={"schema": "attn_sink.CompetitorStructure",
              "note": "N_eff = exp(competitor entropy); top-k over competitor keys only"},
    )


def plot_competitor_structure(cs: CompetitorStructure, out_path: PathLike,
                              show: bool = False) -> Path:
    L = cs.layers.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    ax[0].plot(L, cs.mean_top1, "-o", ms=3, label="top-1")
    ax[0].plot(L, cs.mean_top2, "-o", ms=3, label="top-2")
    ax[0].plot(L, cs.mean_top3, "-o", ms=3, label="top-3")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("competitor logit (nats)")
    ax[0].set_title("(a) strongest competitors", loc="left"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    ax[1].plot(L, cs.mean_neff, "-o", ms=3, color="#8172B3")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("mean $N_{\\mathrm{eff}} = e^{H_{\\mathrm{comp}}}$")
    ax[1].set_title("(b) effective # competitors", loc="left"); ax[1].grid(alpha=0.25)
    axt = ax[1].twinx()
    axt.plot(L, cs.mean_comp_entropy, "--", color="#C44E52", lw=1.1)
    axt.set_ylabel("competitor entropy (nats)", color="#C44E52")
    axt.tick_params(axis="y", labelcolor="#C44E52")

    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].plot(L, cs.r_neff_margin, "-o", ms=3, label="ρ(N_eff, margin)")
    ax[2].plot(L, cs.r_neff_lse, "-o", ms=3, label="ρ(N_eff, LSE)")
    ax[2].plot(L, cs.r_top1_margin, "-o", ms=3, label="ρ(top-1, margin)")
    ax[2].set_ylim(-1.05, 1.05)
    ax[2].set_xlabel("layer"); ax[2].set_ylabel("Pearson r")
    ax[2].set_title("(c) relation to margin / LSE", loc="left"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

    fig.suptitle("Competitor structure across layers", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# 3. LSE decomposition:  LSE = l_max + R,   R = -log p_max^competitor
# ==========================================================================

@dataclass
class LSEDecomposition:
    layers: torch.Tensor
    mean_lmax: torch.Tensor        # E[top-1 competitor logit]
    mean_residual: torch.Tensor    # E[R], R = LSE - l_max >= 0
    mean_lse: torch.Tensor         # = mean_lmax + mean_residual
    mean_pmax: torch.Tensor        # E[exp(-R)] = mean share of the single top competitor
    mean_neff: torch.Tensor        # E[N_eff], for cross-reference
    frac_residual: torch.Tensor    # E[R] / E[LSE-min_floor]... reported as R/(|lmax|+R) proxy
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in
                ("layers", "mean_lmax", "mean_residual", "mean_lse",
                 "mean_pmax", "mean_neff", "frac_residual")}


def lse_decomposition(table: MetricTable) -> LSEDecomposition:
    """Split the competitor log-sum-exp into the strongest competitor and the
    residual competition:

        LSE_comp = l_max + R,      R = log Σ_i exp(l_i - l_max) >= 0.

    R is exactly ``-log p_max^competitor``: the residual is the negative log of
    the single strongest competitor's share of competitor attention mass. So
    R ≈ 0 means one token carries the competition; large R means many moderately
    strong competitors do. That is the question this analysis is posed to answer.

    Needs only comp_max_logit and comp_lse_logit -- both already saved, so this
    is a pure re-read of existing scalars.
    """
    layers = _layers_of(table)
    lay = table["layer"]
    lmax = _col(table, "comp_max_logit")
    lse = _col(table, "comp_lse_logit")
    resid = lse - lmax                       # >= 0 up to float error
    pmax = (-resid).exp()                     # exp(-R) = p_max^competitor
    neff = _col(table, "comp_neff")

    def per_layer(x, s):
        return _nanmean(x[s])

    ml, mr, mlse, mp, mn, fr = [], [], [], [], [], []
    for L in layers:
        s = lay == L
        a, b = per_layer(lmax, s), per_layer(resid, s)
        ml.append(a); mr.append(b); mlse.append(per_layer(lse, s))
        mp.append(per_layer(pmax, s)); mn.append(per_layer(neff, s))
        fr.append(b / (abs(a) + b) if math.isfinite(a) and math.isfinite(b) and (abs(a) + b) > 0 else float("nan"))

    T = lambda x: torch.tensor(x, dtype=torch.float64)
    return LSEDecomposition(
        layers=torch.tensor(layers, dtype=torch.int64),
        mean_lmax=T(ml), mean_residual=T(mr), mean_lse=T(mlse),
        mean_pmax=T(mp), mean_neff=T(mn), frac_residual=T(fr),
        meta={"schema": "attn_sink.LSEDecomposition",
              "identity": "LSE = l_max + R ; R = -log p_max_competitor ; exp(-R)=p_max"},
    )


def plot_lse_decomposition(ld: LSEDecomposition, out_path: PathLike,
                           show: bool = False) -> Path:
    L = ld.layers.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    lmax, resid = ld.mean_lmax.numpy(), ld.mean_residual.numpy()
    ax[0].bar(L, lmax, label="strongest competitor $\\ell_{\\max}$", color="#4C72B0")
    ax[0].bar(L, resid, bottom=lmax, label="residual $R=\\log\\sum e^{\\ell_i-\\ell_{\\max}}$", color="#DD8452")
    ax[0].plot(L, ld.mean_lse, "k_", ms=8, label="LSE (= sum)")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("nats")
    ax[0].set_title("(a) LSE = strongest + residual", loc="left"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    ax[1].plot(L, ld.mean_pmax, "-o", ms=3, color="#C44E52")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("$p_{\\max}^{\\mathrm{comp}} = e^{-R}$")
    ax[1].set_title("(b) dominance of the single strongest competitor", loc="left"); ax[1].grid(alpha=0.25)
    ax[1].text(0.5, 0.92, "→1: one dominant token   →0: many competitors",
               transform=ax[1].transAxes, ha="center", fontsize=8, color="0.35")

    ax[2].plot(L, ld.mean_residual, "-o", ms=3, label="residual R (nats)")
    axt = ax[2].twinx()
    axt.plot(L, ld.mean_neff, "--", color="#8172B3", lw=1.2, label="N_eff")
    axt.set_ylabel("mean N_eff", color="#8172B3"); axt.tick_params(axis="y", labelcolor="#8172B3")
    ax[2].set_xlabel("layer"); ax[2].set_ylabel("residual R (nats)")
    ax[2].set_title("(c) residual vs effective count", loc="left"); ax[2].grid(alpha=0.25)
    ax[2].legend(fontsize=8, loc="upper left")

    fig.suptitle("Competitor LSE decomposition: one dominant token vs many", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# 4. Group statistics: within/between KV-group variance and ICC
# ==========================================================================

def _icc_oneway(values: torch.Tensor, groups: torch.Tensor) -> Tuple[float, float, float]:
    """One-way random-effects ICC(1,1) on head-level means grouped by KV head.

    Returns (within_var, between_var, ICC). Handles unbalanced groups via the
    standard k0 correction; for Qwen3 all groups have n_rep heads so k0 = n_rep.

        MSB = SSB/(G-1),  MSW = SSW/(N-G),  k0 = (N - Σ n_g² / N)/(G-1)
        ICC = (MSB - MSW) / (MSB + (k0-1) MSW)
        within = MSW,  between = (MSB - MSW)/k0

    ICC is the fraction of head-level variance that lies *between* KV groups.
    High ICC  -> heads within a group resemble each other (low specialisation).
    Low/negative ICC -> heads within a group differ as much as across groups
    (consistent with within-group specialisation, the Phase 3.2 question).
    """
    v, g = _finite(values, groups.to(torch.float64))
    if v.numel() < 3:
        return float("nan"), float("nan"), float("nan")
    g = g.long()
    uniq = torch.unique(g)
    G, N = int(uniq.numel()), int(v.numel())
    if G < 2 or N - G < 1:
        return float("nan"), float("nan"), float("nan")
    grand = v.mean()
    ssb = ssw = sum_n2 = 0.0
    for gv in uniq.tolist():
        vg = v[g == gv]
        ng = int(vg.numel())
        mg = vg.mean()
        ssb += ng * float((mg - grand) ** 2)
        ssw += float(((vg - mg) ** 2).sum())
        sum_n2 += ng * ng
    dfb, dfw = G - 1, N - G
    msb, msw = ssb / dfb, ssw / dfw
    k0 = (N - sum_n2 / N) / dfb
    denom = msb + (k0 - 1) * msw
    icc = (msb - msw) / denom if denom != 0 else float("nan")
    within = msw
    between = (msb - msw) / k0 if k0 != 0 else float("nan")
    return within, between, icc


@dataclass
class GroupStats:
    layers: torch.Tensor              # [L]
    metric_names: List[str]
    within: torch.Tensor              # [L, M] within-KV-group variance (MSW)
    between: torch.Tensor             # [L, M] between-KV-group variance
    icc: torch.Tensor                 # [L, M]
    head_means: torch.Tensor          # [L, H, M] per-head mean of each metric (reusable by 3.2)
    kv_group_of_head: torch.Tensor    # [H]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {"layers": self.layers, "within": self.within, "between": self.between,
                "icc": self.icc, "head_means": self.head_means,
                "kv_group_of_head": self.kv_group_of_head}

    def save(self, path: PathLike) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "GroupStats", "metric_names": self.metric_names,
                    "tensors": self.to_tensors(), "meta": self.meta}, path)
        return path


def group_statistics(
    table: MetricTable,
    metrics: Sequence[str] = ("bos_logit", "comp_lse_logit", "margin_lse", "entropy"),
) -> GroupStats:
    """Within/between KV-group variance and ICC per layer, for each metric.

    The unit of analysis is the *head*: each head's value is its mean over
    (query position, sequence), then the ICC is taken over heads grouped by KV
    head. Averaging positions into the head mean is deliberate -- it strips the
    strong position dependence (margin grows with log t) so the ICC reflects
    head *identity*, which is what Phase 3.2 asks about.

    head_means [L, H, M] is saved so Phase 3.2 can recompute any grouping
    statistic it likes without touching the model or the full table.
    """
    layers = _layers_of(table)
    heads = sorted(int(v) for v in torch.unique(table["head"]).tolist())
    H = len(heads)
    head_pos = {h: i for i, h in enumerate(heads)}
    lay, hd, grp = table["layer"], table["head"], table["kv_group"]

    # kv group per head (constant within a head)
    kv_of_head = torch.zeros(H, dtype=torch.int64)
    for h in heads:
        gv = grp[hd == h]
        kv_of_head[head_pos[h]] = int(gv[0]) if gv.numel() else -1

    M = len(metrics)
    head_means = torch.full((len(layers), H, M), torch.nan, dtype=torch.float64)
    within = torch.full((len(layers), M), torch.nan, dtype=torch.float64)
    between = torch.full((len(layers), M), torch.nan, dtype=torch.float64)
    icc = torch.full((len(layers), M), torch.nan, dtype=torch.float64)

    cols = {m: _col(table, m) for m in metrics}
    for li, L in enumerate(layers):
        lmask = lay == L
        for mi, m in enumerate(metrics):
            c = cols[m]
            for h in heads:
                sel = lmask & (hd == h)
                head_means[li, head_pos[h], mi] = _nanmean(c[sel])
            w, b, ic = _icc_oneway(head_means[li, :, mi], kv_of_head)
            within[li, mi] = w; between[li, mi] = b; icc[li, mi] = ic

    return GroupStats(
        layers=torch.tensor(layers, dtype=torch.int64), metric_names=list(metrics),
        within=within, between=between, icc=icc, head_means=head_means,
        kv_group_of_head=kv_of_head,
        meta={"schema": "attn_sink.GroupStats", "n_rep": H // max(1, int(kv_of_head.max()) + 1),
              "icc_definition": "one-way random effects ICC(1,1) on head means",
              "interpretation": "high ICC = heads within a KV group are alike (low "
                                "specialisation); low/negative ICC = within-group "
                                "specialisation"},
    )


def plot_group_statistics(gs: GroupStats, out_path: PathLike, show: bool = False) -> Path:
    L = gs.layers.numpy()
    M = len(gs.metric_names)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    for mi, name in enumerate(gs.metric_names):
        c = cmap(mi)
        ax[0].plot(L, gs.within[:, mi], "-o", ms=3, color=c, label=f"{name} · within")
        ax[0].plot(L, gs.between[:, mi], "--s", ms=3, color=c, alpha=0.7, label=f"{name} · between")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("variance (nats² / nats²)")
    ax[0].set_title("(a) within vs between KV-group variance", loc="left")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=0.25)

    ax[1].axhline(0, color="0.6", lw=0.8)
    for mi, name in enumerate(gs.metric_names):
        ax[1].plot(L, gs.icc[:, mi], "-o", ms=3, color=cmap(mi), label=name)
    ax[1].set_ylim(-0.3, 1.05)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("ICC(1,1) over heads, grouped by KV head")
    ax[1].set_title("(b) intraclass correlation\nhigh = heads in a group alike · low = specialisation",
                    loc="left", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    fig.suptitle("KV-group statistics for Phase 3.2", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# orchestration + persistence
# ==========================================================================

def _save(fig, out_path: PathLike, show: bool) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.show() if show else plt.close(fig)
    return out_path


@dataclass
class ExtendedAnalyses:
    variance: VarianceDecomposition
    competitor: CompetitorStructure
    lse: LSEDecomposition
    group: GroupStats


def run_extended_analyses(
    table: MetricTable,
    out_dir: PathLike,
    group_metrics: Sequence[str] = ("bos_logit", "comp_lse_logit", "margin_lse", "entropy"),
    make_figures: bool = True,
    show: bool = False,
) -> Tuple[ExtendedAnalyses, Dict[str, Path]]:
    """Run all four analyses, save reusable tensors + figures, return both.

    Writes into ``out_dir`` (typically results/phase3/experiment3.1):
        analysis/variance_decomposition.pt
        analysis/competitor_structure.pt
        analysis/lse_decomposition.pt
        analysis/group_statistics.pt      <- the Phase 3.2 hand-off
        analysis/summary.json
        figures/analysis_*.png
    """
    out_dir = Path(out_dir)
    adir = out_dir / "analysis"; adir.mkdir(parents=True, exist_ok=True)
    fdir = out_dir / "figures"; fdir.mkdir(parents=True, exist_ok=True)

    vd = variance_decomposition(table)
    cs = competitor_structure(table)
    ld = lse_decomposition(table)
    gs = group_statistics(table, metrics=group_metrics)

    torch.save({"kind": "VarianceDecomposition", "tensors": vd.to_tensors(), "meta": vd.meta},
               adir / "variance_decomposition.pt")
    torch.save({"kind": "CompetitorStructure", "tensors": cs.to_tensors(), "meta": cs.meta},
               adir / "competitor_structure.pt")
    torch.save({"kind": "LSEDecomposition", "tensors": ld.to_tensors(), "meta": ld.meta},
               adir / "lse_decomposition.pt")
    gs.save(adir / "group_statistics.pt")

    summary = {
        "variance": {"max_identity_error_nats2": vd.max_identity_error,
                     "layers": vd.layers.tolist(),
                     "mean_rho_bos_lse": _nanmean(vd.rho)},
        "competitor": {"mean_neff_by_layer": [round(x, 3) for x in cs.mean_neff.tolist()]},
        "lse": {"mean_pmax_by_layer": [round(x, 3) for x in ld.mean_pmax.tolist()]},
        "group": {"metric_names": gs.metric_names,
                  "icc_last_layer": {n: round(float(gs.icc[-1, i]), 3)
                                     for i, n in enumerate(gs.metric_names)}},
    }
    (adir / "summary.json").write_text(json.dumps(summary, indent=2))

    figs: Dict[str, Path] = {}
    if make_figures:
        figs["variance"] = plot_variance_decomposition(vd, fdir / "analysis_variance_decomposition.png", show)
        figs["competitor"] = plot_competitor_structure(cs, fdir / "analysis_competitor_structure.png", show)
        figs["lse"] = plot_lse_decomposition(ld, fdir / "analysis_lse_decomposition.png", show)
        figs["group"] = plot_group_statistics(gs, fdir / "analysis_group_statistics.png", show)

    return ExtendedAnalyses(vd, cs, ld, gs), figs


def load_group_statistics(path: PathLike, map_location: str = "cpu") -> GroupStats:
    """Phase 3.2 entry point: reload the saved group statistics, no model needed."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("kind") != "GroupStats":
        raise ValueError(f"{path} is not a GroupStats payload")
    t = payload["tensors"]
    return GroupStats(layers=t["layers"], metric_names=payload["metric_names"],
                      within=t["within"], between=t["between"], icc=t["icc"],
                      head_means=t["head_means"], kv_group_of_head=t["kv_group_of_head"],
                      meta=payload["meta"])



# ==========================================================================
# SECTION: prompts
# ==========================================================================
#
# Prompt construction for Phase 3, pinned to the FROZEN En-Vi benchmark.
#
# The 50-pair benchmark (10 each: conversational, news, technical, academic,
# literary) was frozen for Phases 2-4. Phase 3 must therefore draw its prompts
# from it and nowhere else, or every cross-phase comparison silently confounds
# "pre-softmax vs post-softmax" with "different text".
#
# Two rules follow, and both are enforced rather than documented:
#
# 1.  **No fallback corpus.** If the benchmark is not found, this module raises.
#     A silent fallback to embedded or invented prompts is precisely the failure
#     mode we are guarding against -- it would produce plausible-looking figures
#     that are not comparable to Phase 2.
#
# 2.  **No randomness.** Sentences are always ordered by benchmark id. Two runs
#     on the same benchmark produce byte-identical prompts. The SHA-256 of the
#     normalised pair list is recorded in the manifest, so benchmark drift is
#     detectable after the fact rather than invisible.
#
# Length caveat
# -------------
# Benchmark sentences target 15-40 tokens. That is fine for Phase 1/2 sink-score
# work but too short for several Phase 3.1 measurements: the competitor set at
# query position t has only t elements, ``logsumexp`` over competitors cannot show
# its ``log t`` growth across one decade, and entropy is capped at ``log(t+1)``.
#
# ``mode='category_block'`` therefore concatenates the 10 sentences of a category
# in id order into one sequence (~150-400 tokens), the same way Phase 2's 2C built
# its length ladder from benchmark text. Content stays exactly the benchmark;
# length becomes an explicit, documented construction rather than a new
# uncontrolled variable. Every capture records the ``pair_ids`` it was built from.
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, Path]

# The frozen benchmark has 5 categories, 50 pairs total. The fifth is
# 'narrative' in the current corpus; some earlier drafts named it 'literary'.
# We validate the COUNT (EXPECTED_N_CATEGORIES) rather than exact names so either
# spelling loads without a manual patch. KNOWN_CATEGORIES is an allowlist used
# only to flag genuinely unexpected names, and CANONICAL_CATEGORIES is kept as a
# stable, ordered reference (used for deterministic category ordering).
EXPECTED_N_PAIRS = 50
EXPECTED_N_CATEGORIES = 5
KNOWN_CATEGORIES = frozenset(
    {"conversational", "news", "technical", "academic", "narrative", "literary"})
CANONICAL_CATEGORIES = ("conversational", "news", "technical", "academic", "narrative")

# accepted column spellings -> canonical key
_KEYMAP = {
    "id": "id", "pair_id": "id",
    "category": "category", "cat": "category",
    "eng": "eng", "english": "eng", "en": "eng", "source": "eng",
    "vie": "vie", "vietnamese": "vie", "vi": "vie", "target": "vie",
}

_SEARCH_SUBDIRS = ("", "benchmark", "data", "phase2")


class BenchmarkNotFound(FileNotFoundError):
    """Raised instead of falling back to substitute prompts."""


def _normalise_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        rec: Dict[str, Any] = {}
        for k, v in r.items():
            key = _KEYMAP.get(str(k).strip().lower())
            if key:
                rec[key] = v
        if "eng" not in rec or "vie" not in rec:
            raise ValueError(
                f"row {i} lacks an English/Vietnamese column; saw keys {sorted(r)}. "
                f"Accepted spellings: {sorted(set(_KEYMAP))}"
            )
        rec["eng"] = str(rec["eng"]).strip()
        rec["vie"] = str(rec["vie"]).strip()
        rec["category"] = str(rec.get("category", "uncategorised")).strip().lower()
        rec["id"] = int(rec["id"]) if str(rec.get("id", "")).strip().isdigit() else i + 1
        out.append(rec)
    out.sort(key=lambda r: r["id"])
    return out


def benchmark_fingerprint(pairs: Sequence[Dict[str, Any]]) -> str:
    """SHA-256 over the normalised pairs. Goes in the manifest; a change here
    means the benchmark is no longer the one Phase 2 ran on."""
    blob = json.dumps(
        [[p["id"], p["category"], p["eng"], p["vie"]] for p in pairs],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def find_benchmark(search_roots: Sequence[PathLike]) -> Path:
    """Locate the frozen benchmark. CSV is the source of truth, then the
    generated Python forms. Raises rather than substituting anything."""
    names = ("benchmark_en_vi.csv", "benchmark.py", "embedded_pairs.py")
    looked: List[str] = []
    for root in search_roots:
        for sub in _SEARCH_SUBDIRS:
            d = Path(root) / sub if sub else Path(root)
            for n in names:
                p = d / n
                looked.append(str(p))
                if p.exists():
                    return p
    raise BenchmarkNotFound(
        "Frozen En-Vi benchmark not found. Phase 3 refuses to run on substitute "
        "prompts, because that would silently break comparability with Phase 2.\n"
        "Copy benchmark_en_vi.csv (or benchmark.py / embedded_pairs.py) into your "
        "Drive project folder, then re-run.\nLooked in:\n  "
        + "\n  ".join(looked[:24])
    )


def load_benchmark(
    search_roots: Sequence[PathLike],
    strict: bool = True,
) -> Dict[str, Any]:
    """Load and validate the frozen benchmark.

    Returns ``{"pairs": [...], "path": Path, "fingerprint": str, "warnings": [...]}``
    where each pair is ``{"id", "category", "eng", "vie"}``.
    """
    path = find_benchmark(search_roots)

    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pairs = _normalise_rows(rows)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_bm_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                      # type: ignore[union-attr]
        raw = (mod.load_benchmark() if hasattr(mod, "load_benchmark")
               else getattr(mod, "EMBEDDED_PAIRS", None))
        if raw is None:
            raise ValueError(f"{path} exposes neither load_benchmark() nor EMBEDDED_PAIRS")
        pairs = _normalise_rows(raw)

    warnings_: List[str] = []
    if len(pairs) != EXPECTED_N_PAIRS:
        warnings_.append(f"expected {EXPECTED_N_PAIRS} pairs, found {len(pairs)}")
    if len({p["eng"] for p in pairs}) != len(pairs):
        warnings_.append("duplicate English sentences")
    if len({p["vie"] for p in pairs}) != len(pairs):
        warnings_.append("duplicate Vietnamese sentences")
    cats = sorted({p["category"] for p in pairs})
    # Validate category STRUCTURE, not exact names: the frozen benchmark has
    # EXPECTED_N_CATEGORIES distinct categories with EXPECTED_N_PAIRS total. The
    # fifth category is 'narrative' in the current corpus (older drafts called it
    # 'literary'); hard-coding either name is what previously broke loading, so we
    # only check the count and flag names that fall outside the known set.
    if len(cats) != EXPECTED_N_CATEGORIES:
        warnings_.append(
            f"expected {EXPECTED_N_CATEGORIES} categories, found {len(cats)}: {cats}")
    unexpected = [c for c in cats if c not in KNOWN_CATEGORIES]
    if unexpected:
        warnings_.append(
            f"unrecognised categories {unexpected} (known: {sorted(KNOWN_CATEGORIES)})")
    if any(not p["vie"] or not p["eng"] for p in pairs):
        warnings_.append("empty sentence(s) present")
    # diacritics check: the frozen benchmark uses full Vietnamese diacritics.
    # Its absence means an older placeholder corpus, which tokenises differently.
    if not any(re.search(r"[\u00C0-\u1EF9]", p["vie"]) for p in pairs):
        warnings_.append(
            "no Vietnamese diacritics detected -- this looks like the OLD "
            "placeholder corpus, not the frozen benchmark"
        )

    if strict and warnings_:
        raise ValueError(
            "benchmark failed validation:\n  - " + "\n  - ".join(warnings_)
            + f"\n(loaded from {path}). Pass strict=False to proceed anyway."
        )

    return {"pairs": pairs, "path": path, "fingerprint": benchmark_fingerprint(pairs),
            "warnings": warnings_}


# --------------------------------------------------------------------------
# Deterministic prompt construction
# --------------------------------------------------------------------------

@dataclass
class Prompt:
    seq_id: str
    text: str
    language: str          # 'eng' | 'vie'
    category: str          # benchmark category, or 'all' for full_block
    pair_ids: List[int]    # exact benchmark rows this was built from
    mode: str

    def as_meta(self) -> Dict[str, Any]:
        return {"category": self.category, "language": self.language,
                "pair_ids": self.pair_ids, "prompt_mode": self.mode}


def build_prompts(
    pairs: Sequence[Dict[str, Any]],
    mode: str = "category_block",
    languages: Sequence[str] = ("eng", "vie"),
    categories: Optional[Sequence[str]] = None,
    joiner: str = " ",
    max_sequences: Optional[int] = None,
) -> List[Prompt]:
    """Build Phase 3 prompts from the frozen benchmark. Fully deterministic.

    Modes
    -----
    ``per_sentence``    one sequence per (pair, language). T ~ 15-40 tokens.
                        Maximum comparability with Phase 2's per-prompt unit,
                        but too short for the position-dependent measurements.
    ``category_block``  the 10 sentences of a category, in id order, joined into
                        one sequence per (category, language). T ~ 150-400.
                        Default: the shortest construction that still lets the
                        log-t behaviour of the competitor term be visible.
    ``full_block``      all 50 in id order, one sequence per language.
    """
    bad_langs = [l for l in languages if l not in ("eng", "vie")]
    if bad_langs:
        raise ValueError(f"languages must be 'eng' and/or 'vie'; got {bad_langs}")

    wanted = list(categories) if categories else None
    sel = [p for p in pairs if wanted is None or p["category"] in wanted]
    if not sel:
        raise ValueError(f"no benchmark pairs match categories={categories}")
    sel = sorted(sel, key=lambda p: p["id"])          # determinism

    out: List[Prompt] = []
    if mode == "per_sentence":
        for lang in languages:
            for p in sel:
                out.append(Prompt("", p[lang], lang, p["category"], [p["id"]], mode))
    elif mode == "category_block":
        cats = [c for c in CANONICAL_CATEGORIES if any(p["category"] == c for p in sel)]
        cats += sorted({p["category"] for p in sel} - set(cats))
        for lang in languages:
            for c in cats:
                grp = [p for p in sel if p["category"] == c]
                out.append(Prompt("", joiner.join(p[lang] for p in grp), lang, c,
                                  [p["id"] for p in grp], mode))
    elif mode == "full_block":
        for lang in languages:
            out.append(Prompt("", joiner.join(p[lang] for p in sel), lang, "all",
                              [p["id"] for p in sel], mode))
    else:
        raise ValueError(f"unknown mode {mode!r}; expected per_sentence, "
                         f"category_block or full_block")

    if max_sequences is not None:
        out = out[:max_sequences]
    for i, pr in enumerate(out):
        pr.seq_id = f"seq{i:03d}"
    return out


def describe_prompts(prompts: Sequence[Prompt], tokenizer: Any = None,
                     max_len: Optional[int] = None) -> str:
    """Human-readable table. With a tokenizer, reports real token lengths --
    worth looking at before committing, since T drives storage quadratically."""
    lines = [f"{len(prompts)} sequences  (mode={prompts[0].mode if prompts else '-'})",
             f"  {'seq':<8}{'lang':<6}{'category':<16}{'pairs':>6}{'chars':>8}"
             + (f"{'tokens':>8}" if tokenizer is not None else "")]
    tls: List[int] = []
    for p in prompts:
        row = (f"  {p.seq_id:<8}{p.language:<6}{p.category:<16}"
               f"{len(p.pair_ids):>6}{len(p.text):>8}")
        if tokenizer is not None:
            n = len(tokenizer(p.text, add_special_tokens=False)["input_ids"])
            tls.append(n)
            flag = "  (truncated)" if max_len and n > max_len else ""
            row += f"{n:>8}{flag}"
        lines.append(row)
    if tls:
        lines.append(f"  token length: min {min(tls)}  median "
                     f"{sorted(tls)[len(tls)//2]}  max {max(tls)}")
        if max_len:
            eff = [min(t + 1, max_len) for t in tls]
            lines.append(f"  effective T after MAX_LEN={max_len} and BOS: "
                         f"min {min(eff)}  max {max(eff)}")
        if max(tls) < 64:
            lines.append("  [warn] sequences are short: the competitor set at position t "
                         "has only t members, so the log-t growth of the competitor "
                         "logsumexp will not be visible. Consider category_block.")
    return "\n".join(lines)
