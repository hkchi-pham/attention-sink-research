"""
Phase 3.2 -- Origin of within-KV-group specialization  (OBSERVATIONAL ONLY)

Phase 3.1 established that heads sharing a KV head are more alike than heads in
different groups, yet substantial within-group variation survives, growing with
depth (the ICC in Part 12 drops in later layers). Phase 3.2 asks *where in the
attention computation* that within-group difference first appears -- and does so
without touching the model. Everything here reads Phase 3.1's saved artifacts:

    captures/seq*.pt      q [L,H,T,D] (post-RoPE), k [L,G,T,D], scaling, mapping
    metrics/table.pt      pooled per-(layer,head,pos) scalars
    analysis/group_statistics.pt   head means + ICC + kv_group_of_head

Because within a KV group K and V are identical, any divergence must originate in
the *query* path. The four analyses localise the stage:

    1. query_norm_stats        do the heads' query vectors differ in magnitude?
    2. score_correlations       do the raw QK^T matrices differ before softmax?
    3. bos_competition          is a sink head favouring BOS or suppressing rivals?
    4. divergence_through_depth  do heads start identical and drift apart with depth?

Nothing here modifies weights, patches activations, or ablates -- those are
Phase 3.3. This module only measures.

Design notes
------------
* Reuses the Phase 3.1 loaders (`phase3_utils`) verbatim; no re-inference.
* The raw QK^T matrix is recomputed from the stored q and k one layer at a time,
  so peak memory is [H,T,T] rather than [L,H,T,T]. The recompute is byte-identical
  to what Phase 3.1 validated (check C2), and if the capture happens to still hold
  `logits` we assert agreement.
* Score correlations are computed once and shared by analyses 2 and 4.
* All statistics are float64 and NaN-aware.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
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
from matplotlib import cm
from matplotlib.colors import Normalize

import phase3_utils as U

PathLike = Union[str, Path]

# A consistent, colour-blind-friendly palette for KV groups across every figure.
_GROUP_CMAP = "tab20"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ==========================================================================
# Loading Phase 3.1 outputs  (the "reuse the pipeline" requirement)
# ==========================================================================

@dataclass
class Phase32Inputs:
    """Everything Phase 3.2 needs, loaded once from Phase 3.1 artefacts."""
    exp_dir: Path
    captures: List[Any]                 # list[CaptureResult]
    table: Any                          # MetricTable
    group_stats: Optional[Any]          # GroupStats or None
    n_layers: int
    n_heads: int
    n_rep: int
    n_groups: int
    kv_group_of_head: torch.Tensor      # [H]
    layers: List[int]

    def group_members(self, g: int) -> List[int]:
        return [h for h in range(self.n_heads) if int(self.kv_group_of_head[h]) == g]

    def head_pairs(self) -> List[Tuple[int, int]]:
        """All within-group head pairs (one pair per group when n_rep == 2)."""
        pairs: List[Tuple[int, int]] = []
        for g in range(self.n_groups):
            pairs.extend(combinations(self.group_members(g), 2))
        return pairs


def load_phase32_inputs(exp_dir: PathLike, load_captures: bool = True) -> Phase32Inputs:
    """Load Phase 3.1 outputs. No model, no inference.

    ``exp_dir`` is the Phase 3.1 experiment directory, e.g.
    results/phase3/experiment3.1.
    """
    exp_dir = Path(exp_dir)
    table = U.load_metric_table(exp_dir / "metrics" / "table.pt")

    gs_path = exp_dir / "analysis" / "group_statistics.pt"
    group_stats = U.load_group_statistics(gs_path) if gs_path.exists() else None

    caps: List[Any] = []
    if load_captures:
        for p in sorted((exp_dir / "captures").glob("*.pt")):
            caps.append(U.load_capture(p))
        if not caps:
            raise FileNotFoundError(
                f"no captures under {exp_dir/'captures'} -- Phase 3.2 needs the "
                "stored q/k vectors for analyses 2 and 4.")

    # topology from the table (always present) or the first capture
    kv = None
    if group_stats is not None:
        kv = group_stats.kv_group_of_head.clone()
    elif caps:
        kv = caps[0].kv_group_of_head.clone()
    else:
        H = int(table["head"].max()) + 1
        kv = torch.tensor([int(table.where(head=h)["kv_group"][0]) for h in range(H)])
    H = int(kv.numel())
    n_groups = int(kv.max()) + 1
    n_rep = H // n_groups
    layers = sorted(int(v) for v in torch.unique(table["layer"]).tolist())

    return Phase32Inputs(
        exp_dir=exp_dir, captures=caps, table=table, group_stats=group_stats,
        n_layers=len(layers), n_heads=H, n_rep=n_rep, n_groups=n_groups,
        kv_group_of_head=kv, layers=layers,
    )


# ==========================================================================
# Shared numeric helpers
# ==========================================================================

def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Population Pearson correlation, NaN-aware, guards zero variance."""
    keep = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[keep].double(), y[keep].double()
    if x.numel() < 2:
        return float("nan")
    sx, sy = x.std(unbiased=False), y.std(unbiased=False)
    if float(sx) == 0.0 or float(sy) == 0.0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity of two [N, D] tensors -> [N]."""
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 0, num / den.clamp_min(1e-30), torch.full_like(num, float("nan")))


def recompute_layer_logits(cap: Any, layer_pos: int, dtype=torch.float64) -> torch.Tensor:
    """Raw QK^T for one captured layer -> [H, T, T], unmasked.

    K is expanded from per-KV-head to per-query-head, so within a KV group the
    key operand is identical across heads and any difference in the returned
    matrix comes purely from the query projection. Byte-identical to Phase 3.1's
    check C2. Kept to one layer at a time to bound memory at [H,T,T].
    """
    q = cap.queries[layer_pos].to(dtype)                      # [H,T,D]
    k = cap.keys[layer_pos].to(dtype)                         # [G,T,D]
    k_h = k.index_select(0, cap.kv_group_of_head)             # [H,T,D]
    scaling = float(cap.meta["scaling"])
    logits = torch.einsum("htd,hsd->hts", q, k_h) * scaling   # [H,T,T]
    if cap.logits is not None:                                 # verify if available
        ref = cap.logits[layer_pos].to(dtype)
        err = (logits - ref).abs().max().item()
        if err > 1e-4:
            raise AssertionError(f"recomputed logits disagree with stored ({err:.2e})")
    return logits


def _causal_tri_index(T: int, min_query_pos: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row/col indices of the strictly-causal region used for matrix correlations:
    keys s <= t, queries t >= min_query_pos. Excludes the BOS-as-query row."""
    t = torch.arange(T)
    mask = (t[:, None] >= t[None, :]) & (t[:, None] >= min_query_pos)
    return mask.nonzero(as_tuple=True)


# ==========================================================================
# Per-head table aggregates  (feeds Analyses 1 & 3 and the CSV; no captures)
# ==========================================================================

@dataclass
class HeadTable:
    """Per-(layer, head) means pooled over tokens and sequences."""
    layer: np.ndarray
    head: np.ndarray
    kv_group: np.ndarray
    query_norm: np.ndarray
    bos_logit: np.ndarray
    largest_competitor: np.ndarray      # comp_max_logit
    bos_advantage: np.ndarray           # margin_max = bos_logit - comp_max_logit
    sink_score: np.ndarray              # mean bos_prob

    def as_rows(self) -> List[Dict[str, Any]]:
        n = len(self.layer)
        keys = ["layer", "head", "kv_group", "query_norm", "bos_logit",
                "largest_competitor", "bos_advantage", "sink_score"]
        out = []
        for i in range(n):
            out.append({k: (int(getattr(self, k)[i]) if k in ("layer", "head", "kv_group")
                            else float(getattr(self, k)[i])) for k in keys})
        return out


def build_head_table(inp: Phase32Inputs) -> HeadTable:
    """Aggregate the pooled MetricTable to one row per (layer, head).

    Uses only scalars Phase 3.1 already saved -- no captures, no recompute.
    bos_advantage is exactly margin_max; sink_score is mean bos_prob.
    """
    t = inp.table
    layer_col, head_col = t["layer"], t["head"]
    rows_layer, rows_head, rows_grp = [], [], []
    qn, bl, lc, adv, ss = [], [], [], [], []

    def m(col, sel):
        v = t[col][sel].double()
        v = v[torch.isfinite(v)]
        return float(v.mean()) if v.numel() else float("nan")

    for L in inp.layers:
        for h in range(inp.n_heads):
            sel = (layer_col == L) & (head_col == h)
            if not bool(sel.any()):
                continue
            rows_layer.append(L); rows_head.append(h)
            rows_grp.append(int(inp.kv_group_of_head[h]))
            qn.append(m("query_norm", sel))
            bl.append(m("bos_logit", sel))
            lc.append(m("comp_max_logit", sel))
            adv.append(m("margin_max", sel))
            ss.append(m("bos_prob", sel))

    arr = lambda x: np.asarray(x, dtype=float)
    return HeadTable(
        layer=np.asarray(rows_layer, dtype=int), head=np.asarray(rows_head, dtype=int),
        kv_group=np.asarray(rows_grp, dtype=int), query_norm=arr(qn), bos_logit=arr(bl),
        largest_competitor=arr(lc), bos_advantage=arr(adv), sink_score=arr(ss),
    )


# ==========================================================================
# Analysis 1 -- Query vector statistics
# ==========================================================================

@dataclass
class QueryNormStats:
    layers: List[int]
    n_heads: int
    kv_group_of_head: torch.Tensor
    per_head_mean: np.ndarray     # [L, H]
    per_head_std: np.ndarray      # [L, H] (over tokens+seqs)
    within_group_spread: np.ndarray   # [L, G] max-min of per-head means in a group
    global_min: float
    global_max: float
    meta: Dict[str, Any] = field(default_factory=dict)


def query_norm_stats(inp: Phase32Inputs, head_tbl: Optional[HeadTable] = None) -> QueryNormStats:
    """Per-head query L2 norm, mean and std over tokens, plus within-group spread.

    Answers: do sink heads simply emit larger query vectors? In Qwen3 the query
    passes through QK-RMSNorm before RoPE (norm-preserving), so ||q|| is confined
    to a shared band [sqrt(D)*min|gamma|, sqrt(D)*max|gamma|] identical across
    heads. Any within-group spread therefore reflects how each head's query
    *directions* land in that band, not free magnitude scaling.
    """
    t = inp.table
    L, H, G = inp.n_layers, inp.n_heads, inp.n_groups
    lay, hd = t["layer"], t["head"]
    qn = t["query_norm"].double()

    mean = np.full((L, H), np.nan)
    std = np.full((L, H), np.nan)
    for li, Lx in enumerate(inp.layers):
        for h in range(H):
            v = qn[(lay == Lx) & (hd == h)]
            v = v[torch.isfinite(v)]
            if v.numel():
                mean[li, h] = float(v.mean()); std[li, h] = float(v.std(unbiased=False))

    spread = np.full((L, G), np.nan)
    for li in range(L):
        for g in range(G):
            members = inp.group_members(g)
            vals = mean[li, members]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                spread[li, g] = float(vals.max() - vals.min())

    finite = mean[np.isfinite(mean)]
    return QueryNormStats(
        layers=list(inp.layers), n_heads=H, kv_group_of_head=inp.kv_group_of_head,
        per_head_mean=mean, per_head_std=std, within_group_spread=spread,
        global_min=float(finite.min()) if finite.size else float("nan"),
        global_max=float(finite.max()) if finite.size else float("nan"),
        meta={"note": "||q|| is post-RoPE, post-QK-norm; band is shared across heads"},
    )


def plot_query_norms(qns: QueryNormStats, out_path: PathLike, show: bool = False) -> Path:
    L, H, G = qns.n_heads and len(qns.layers), qns.n_heads, int(qns.kv_group_of_head.max()) + 1
    cmap = plt.get_cmap(_GROUP_CMAP)
    colors = [cmap(int(qns.kv_group_of_head[h]) % 20) for h in range(H)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    # (a) boxplot of per-layer query norm for each head, coloured by KV group
    data = [qns.per_head_mean[:, h][np.isfinite(qns.per_head_mean[:, h])] for h in range(H)]
    bp = ax[0].boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax[0].set_xlabel("query head"); ax[0].set_ylabel("mean $\\|q\\|_2$ (over tokens, per layer)")
    ax[0].set_title("(a) query-norm distribution per head, coloured by KV group", loc="left")
    ax[0].set_xticks(range(1, H + 1)); ax[0].set_xticklabels(range(H), fontsize=7)
    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=cmap(g % 20),
               markeredgecolor="none", label=f"group {g}") for g in range(G)]
    ax[0].legend(handles=handles, ncol=max(1, G // 4), fontsize=7, title="KV group")

    # (b) within-group spread of the head means, across depth
    for g in range(G):
        ax[1].plot(qns.layers, qns.within_group_spread[:, g], "-o", ms=3,
                   color=cmap(g % 20), label=f"group {g}")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("within-group $\\|q\\|$ spread (max−min)")
    ax[1].set_title("(b) how far apart are query magnitudes inside a KV group?", loc="left")
    ax[1].legend(ncol=max(1, G // 4), fontsize=7)

    fig.suptitle(f"Analysis 1 · Query vector magnitude   "
                 f"(band [{qns.global_min:.2f}, {qns.global_max:.2f}])", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# Analysis 2 & 4 shared core -- within-group similarity from the q/k vectors
# ==========================================================================

@dataclass
class WithinGroupSimilarity:
    """Per (layer, head-pair) similarity, averaged over sequences.

    pair_groups[i] is the KV group of pair i; pairs[i] = (h, h').
    Each array is [L, P] where P = number of within-group pairs.
    """
    layers: List[int]
    pairs: List[Tuple[int, int]]
    pair_groups: List[int]
    query_cosine: np.ndarray        # mean cos(q_h, q_h') over tokens
    score_corr: np.ndarray          # Pearson corr of causal QK^T
    bos_logit_corr: np.ndarray      # Pearson corr of BOS logits over tokens
    meta: Dict[str, Any] = field(default_factory=dict)


def within_group_similarity(inp: Phase32Inputs, min_query_pos: int = 1) -> WithinGroupSimilarity:
    """Core reused by analyses 2 and 4. For every within-group head pair and
    every layer, average three similarities over the captured sequences:

      * query_cosine   : mean over tokens of cos(q_h[t], q_h'[t])
      * score_corr     : Pearson corr of the causal QK^T entries (K is shared,
                         so this isolates the query-projection difference)
      * bos_logit_corr : Pearson corr of the BOS-column logits across tokens

    Computed one layer at a time to keep memory at [H,T,T]. This is the single
    place QK^T is recomputed; analyses 2 and 4 both read the result.
    """
    pairs = inp.head_pairs()
    pair_groups = [int(inp.kv_group_of_head[a]) for a, _ in pairs]
    P, L = len(pairs), inp.n_layers
    cos_acc = np.zeros((L, P)); cos_n = np.zeros((L, P))
    scr_acc = np.zeros((L, P)); scr_n = np.zeros((L, P))
    bos_acc = np.zeros((L, P)); bos_n = np.zeros((L, P))

    for cap in inp.captures:
        T = cap.seq_len
        ri, ci = _causal_tri_index(T, min_query_pos)
        for li in range(cap.n_layers):
            Lg = recompute_layer_logits(cap, li)          # [H,T,T]
            q = cap.queries[li].double()                   # [H,T,D]
            bos_col = Lg[:, :, 0]                           # [H,T] BOS logits per query
            for pi, (a, b) in enumerate(pairs):
                # query cosine over tokens (positions >= min_query_pos)
                cos = _cosine_rows(q[a, min_query_pos:], q[b, min_query_pos:])
                cos = cos[torch.isfinite(cos)]
                if cos.numel():
                    cos_acc[li, pi] += float(cos.mean()); cos_n[li, pi] += 1
                # raw score-matrix correlation over the causal region
                sc = _pearson(Lg[a][ri, ci], Lg[b][ri, ci])
                if math.isfinite(sc):
                    scr_acc[li, pi] += sc; scr_n[li, pi] += 1
                # BOS-logit correlation over query tokens
                bc = _pearson(bos_col[a, min_query_pos:], bos_col[b, min_query_pos:])
                if math.isfinite(bc):
                    bos_acc[li, pi] += bc; bos_n[li, pi] += 1

    def avg(acc, n):
        out = np.full_like(acc, np.nan)
        nz = n > 0
        out[nz] = acc[nz] / n[nz]
        return out

    return WithinGroupSimilarity(
        layers=list(inp.layers), pairs=pairs, pair_groups=pair_groups,
        query_cosine=avg(cos_acc, cos_n), score_corr=avg(scr_acc, scr_n),
        bos_logit_corr=avg(bos_acc, bos_n),
        meta={"min_query_pos": min_query_pos, "n_sequences": len(inp.captures)},
    )


def compatibility_example(inp: Phase32Inputs, layer_pos: int, group: int,
                          min_query_pos: int = 0) -> Dict[str, Any]:
    """Return two heads' causal QK^T matrices and their difference for heatmaps.

    Picks the first sequence and the first within-group pair of ``group``.
    """
    cap = inp.captures[0]
    members = inp.group_members(group)
    if len(members) < 2:
        raise ValueError(f"group {group} has <2 heads")
    a, b = members[0], members[1]
    Lg = recompute_layer_logits(cap, layer_pos)            # [H,T,T]
    T = cap.seq_len
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    ma = Lg[a].masked_fill(~causal, float("nan")).numpy()
    mb = Lg[b].masked_fill(~causal, float("nan")).numpy()
    return {"layer": int(cap.layer_index[layer_pos]), "group": group,
            "head_a": a, "head_b": b, "mat_a": ma, "mat_b": mb, "diff": ma - mb}


def plot_compatibility(inp: Phase32Inputs, sim: WithinGroupSimilarity,
                       out_path: PathLike, layer_positions: Optional[Sequence[int]] = None,
                       group: int = 0, show: bool = False) -> Path:
    """Heatmaps of QK^T for a shared-KV head pair at early/mid/late layers, their
    differences, and the within-group score correlation across depth."""
    if layer_positions is None:
        Lp = inp.captures[0].n_layers
        layer_positions = [0, Lp // 2, Lp - 1]
    ncol = len(layer_positions)
    fig = plt.figure(figsize=(5 * ncol, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1, 1, 0.9])

    for j, lp in enumerate(layer_positions):
        ex = compatibility_example(inp, lp, group)
        vmax = np.nanmax(np.abs(np.stack([ex["mat_a"], ex["mat_b"]])))
        for row, key, title in [(0, "mat_a", f"head {ex['head_a']}"),
                                 (1, "mat_b", f"head {ex['head_b']}")]:
            ax = fig.add_subplot(gs[row, j])
            im = ax.imshow(ex[key], cmap="viridis", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"L{ex['layer']} · {title}", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("query t")
            fig.colorbar(im, ax=ax, fraction=0.046)
        axd = fig.add_subplot(gs[2, j])
        dmax = np.nanmax(np.abs(ex["diff"]))
        im = axd.imshow(ex["diff"], cmap="coolwarm", vmin=-dmax, vmax=dmax, aspect="auto")
        axd.set_title(f"L{ex['layer']} · difference (same K)", fontsize=9, loc="left")
        axd.set_xlabel("key s")
        if j == 0:
            axd.set_ylabel("query t")
        fig.colorbar(im, ax=axd, fraction=0.046)

    fig.suptitle(f"Analysis 2 · Raw QK$^\\top$ for a shared-KV pair (group {group}) "
                 f"— identical K, so any difference is the query", fontsize=12)
    p = _save(fig, out_path, show)

    # companion figure: score correlation across depth, all groups
    fig2, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cmap = plt.get_cmap(_GROUP_CMAP)
    for pi, (a, b) in enumerate(sim.pairs):
        g = sim.pair_groups[pi]
        ax.plot(sim.layers, sim.score_corr[:, pi], "-o", ms=3, color=cmap(g % 20),
                label=f"group {g} ({a},{b})")
    ax.set_xlabel("layer"); ax.set_ylabel("Pearson corr of causal QK$^\\top$ (within group)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Analysis 2 · do shared-KV heads already differ before softmax?", loc="left")
    ax.legend(ncol=2, fontsize=7)
    _save(fig2, Path(out_path).with_name(Path(out_path).stem + "_score_corr.png"), show)
    return p


# ==========================================================================
# Analysis 3 -- BOS competition
# ==========================================================================

@dataclass
class BOSCompetition:
    head_tbl: HeadTable
    sink_score_by_head: np.ndarray      # [H] mean sink score across layers
    sink_threshold: float
    sink_heads: List[int]
    nonsink_heads: List[int]
    adv_sink: np.ndarray                # per-(layer,head) bos_advantage, sink heads
    adv_nonsink: np.ndarray
    bos_logit_sink_mean: float
    bos_logit_nonsink_mean: float
    competitor_sink_mean: float
    competitor_nonsink_mean: float
    meta: Dict[str, Any] = field(default_factory=dict)


def bos_competition(inp: Phase32Inputs, head_tbl: HeadTable,
                    sink_quantile: float = 0.75) -> BOSCompetition:
    """Split heads into sink / non-sink by their sink score (mean BOS prob), then
    ask whether the sink heads win by *favouring BOS* (higher BOS logit) or by
    *suppressing rivals* (lower competitor logit).

    bos_advantage = bos_logit - largest_competitor is exactly margin_max.
    """
    H = inp.n_heads
    sink_by_head = np.array([np.nanmean(head_tbl.sink_score[head_tbl.head == h]) for h in range(H)])
    thr = float(np.nanquantile(sink_by_head, sink_quantile))
    sink_heads = [h for h in range(H) if sink_by_head[h] >= thr]
    nonsink = [h for h in range(H) if h not in sink_heads]

    in_sink = np.isin(head_tbl.head, sink_heads)
    adv_sink = head_tbl.bos_advantage[in_sink]
    adv_nonsink = head_tbl.bos_advantage[~in_sink]

    def mean_where(col, mask):
        v = col[mask]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    return BOSCompetition(
        head_tbl=head_tbl, sink_score_by_head=sink_by_head, sink_threshold=thr,
        sink_heads=sink_heads, nonsink_heads=nonsink,
        adv_sink=adv_sink, adv_nonsink=adv_nonsink,
        bos_logit_sink_mean=mean_where(head_tbl.bos_logit, in_sink),
        bos_logit_nonsink_mean=mean_where(head_tbl.bos_logit, ~in_sink),
        competitor_sink_mean=mean_where(head_tbl.largest_competitor, in_sink),
        competitor_nonsink_mean=mean_where(head_tbl.largest_competitor, ~in_sink),
        meta={"sink_quantile": sink_quantile},
    )


def plot_bos_competition(bc: BOSCompetition, out_path: PathLike, show: bool = False) -> Path:
    ht = bc.head_tbl
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # (a) BOS advantage distribution, sink vs non-sink
    ax[0].hist(bc.adv_nonsink[np.isfinite(bc.adv_nonsink)], bins=30, alpha=0.6,
               label="non-sink heads", color="#8899AA", density=True)
    ax[0].hist(bc.adv_sink[np.isfinite(bc.adv_sink)], bins=30, alpha=0.6,
               label="sink heads", color="#CC4444", density=True)
    ax[0].axvline(0, color="0.4", lw=0.8)
    ax[0].set_xlabel("BOS advantage = BOS logit − largest competitor (nats)")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) do sink heads run a larger BOS advantage?", loc="left")
    ax[0].legend()

    # (b) favour vs suppress: grouped means
    labels = ["BOS logit", "largest competitor"]
    x = np.arange(2); w = 0.35
    ax[1].bar(x - w / 2, [bc.bos_logit_sink_mean, bc.competitor_sink_mean], w,
              label="sink", color="#CC4444")
    ax[1].bar(x + w / 2, [bc.bos_logit_nonsink_mean, bc.competitor_nonsink_mean], w,
              label="non-sink", color="#8899AA")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean logit (nats)")
    ax[1].set_title("(b) favour BOS, or suppress rivals?", loc="left")
    ax[1].legend()

    # (c) BOS advantage vs sink score, per head
    sc = np.array([bc.sink_score_by_head[h] for h in ht.head])
    ax[2].scatter(sc, ht.bos_advantage, s=10, c=["#CC4444" if h in bc.sink_heads else "#8899AA"
                                                 for h in ht.head], alpha=0.6)
    ax[2].axvline(bc.sink_threshold, color="0.4", ls="--", lw=0.8, label="sink threshold")
    ax[2].set_xlabel("head sink score (mean BOS prob)")
    ax[2].set_ylabel("BOS advantage (nats)")
    ax[2].set_title("(c) advantage tracks sink score", loc="left")
    ax[2].legend()

    fig.suptitle("Analysis 3 · BOS competition", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# Analysis 4 -- Divergence through depth
# ==========================================================================

def plot_divergence(sim: WithinGroupSimilarity, out_path: PathLike, show: bool = False) -> Path:
    """Three within-group similarity metrics vs depth, one line per group."""
    cmap = plt.get_cmap(_GROUP_CMAP)
    panels = [("query_cosine", "cosine of query vectors", sim.query_cosine),
              ("score_corr", "corr of raw QK$^\\top$", sim.score_corr),
              ("bos_logit_corr", "corr of BOS logits", sim.bos_logit_corr)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for k, (name, ylabel, data) in enumerate(panels):
        for pi in range(len(sim.pairs)):
            g = sim.pair_groups[pi]
            ax[k].plot(sim.layers, data[:, pi], "-o", ms=3, color=cmap(g % 20), alpha=0.85)
        # bold mean across groups
        ax[k].plot(sim.layers, np.nanmean(data, axis=1), "-", color="black", lw=2.2,
                   label="mean over groups")
        ax[k].set_xlabel("layer"); ax[k].set_ylabel(f"within-group {ylabel}")
        ax[k].set_ylim(-0.05, 1.05)
        ax[k].set_title(f"({'abc'[k]}) {ylabel}", loc="left")
        ax[k].legend()
    fig.suptitle("Analysis 4 · Where do shared-KV heads diverge through depth?", fontsize=12)
    return _save(fig, out_path, show)


# ==========================================================================
# Summary CSV + interpretation + orchestration
# ==========================================================================

def write_summary_csv(head_tbl: HeadTable, path: PathLike) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = head_tbl.as_rows()
    fields = ["layer", "head", "kv_group", "query_norm", "bos_logit",
              "largest_competitor", "bos_advantage", "sink_score"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _trend(y: np.ndarray, x: Optional[Sequence[int]] = None) -> float:
    """Sign/size of a linear trend: slope per layer, NaN-aware."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def interpret(qns: QueryNormStats, sim: WithinGroupSimilarity,
              bc: BOSCompetition) -> Dict[str, str]:
    """Data-driven observations. Each string is chosen from the measurements, not
    hard-coded conclusions."""
    obs: Dict[str, str] = {}

    # A1: how big is within-group query-norm spread relative to the band?
    band = qns.global_max - qns.global_min
    spread = float(np.nanmean(qns.within_group_spread))
    rel = spread / band if band > 1e-9 else 0.0
    small = (band <= 1e-6) or (rel < 0.25)
    obs["analysis1"] = (
        f"Within a KV group, mean query-norm spread is {spread:.3f} "
        f"({rel:.0%} of the {qns.global_min:.2f}–{qns.global_max:.2f} band). "
        + ("Query magnitudes are effectively equal within a group, so specialization "
           "is not driven by vector magnitude."
           if small else
           "Query magnitudes differ appreciably within a group, so magnitude is one "
           "contributing factor."))

    # A2: are raw scores already different before softmax?
    sc_mean = float(np.nanmean(sim.score_corr))
    obs["analysis2"] = (
        f"Mean within-group correlation of the raw QK^T matrices is {sc_mean:.2f}. "
        + ("Despite sharing identical Keys, the heads' pre-softmax compatibility "
           "matrices already differ substantially — divergence originates in the "
           "query projection, before masking or softmax."
           if sc_mean < 0.9 else
           "The pre-softmax matrices are highly aligned; divergence is not yet "
           "present at the raw-score stage."))

    # A3: favour BOS vs suppress rivals
    d_bos = bc.bos_logit_sink_mean - bc.bos_logit_nonsink_mean
    d_comp = bc.competitor_sink_mean - bc.competitor_nonsink_mean
    adv_s = float(np.nanmean(bc.adv_sink)); adv_n = float(np.nanmean(bc.adv_nonsink))
    driver = ("stronger BOS compatibility" if abs(d_bos) >= abs(d_comp)
              else "suppression of competing tokens")
    obs["analysis3"] = (
        f"Sink heads carry a larger BOS advantage ({adv_s:+.2f} vs {adv_n:+.2f} nats). "
        f"Relative to non-sink heads their BOS logit shifts {d_bos:+.2f} and their "
        f"largest competitor shifts {d_comp:+.2f}, so the advantage is driven mainly by "
        f"{driver}.")

    # A4: diverge through depth?
    slope = _trend(np.nanmean(sim.query_cosine, axis=1), sim.layers)
    early = float(np.nanmean(sim.query_cosine[0]))
    late = float(np.nanmean(sim.query_cosine[-1]))
    obs["analysis4"] = (
        f"Mean within-group query cosine goes {early:.2f} (first layer) → {late:.2f} "
        f"(last), slope {slope:+.3f}/layer. "
        + ("Heads begin nearly identical and progressively diverge with depth — "
           "specialization develops during representation refinement rather than "
           "existing from the start."
           if slope < -0.005 and early > late else
           "Heads do not show a clear monotone drift; specialization is already "
           "present in early layers or is not depth-ordered."))
    return obs


@dataclass
class Phase32Results:
    inputs: Phase32Inputs
    head_tbl: HeadTable
    qnorm: QueryNormStats
    similarity: WithinGroupSimilarity
    bos: BOSCompetition
    observations: Dict[str, str]
    figures: Dict[str, Path]
    csv_path: Path


def run_phase32(exp_dir: PathLike, out_dir: PathLike = "phase3_2_origin",
                sink_quantile: float = 0.75, make_figures: bool = True,
                show: bool = False) -> Phase32Results:
    """Run all four observational analyses from Phase 3.1 artefacts.

    Writes summary.csv, four figures, and intermediate tensors into ``out_dir``.
    No model is loaded.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tensors").mkdir(parents=True, exist_ok=True)

    inp = load_phase32_inputs(exp_dir)
    head_tbl = build_head_table(inp)
    csv_path = write_summary_csv(head_tbl, out_dir / "summary.csv")

    qns = query_norm_stats(inp, head_tbl)
    sim = within_group_similarity(inp)             # shared by analyses 2 & 4
    bc = bos_competition(inp, head_tbl, sink_quantile=sink_quantile)
    obs = interpret(qns, sim, bc)

    # persist intermediate tensors for any later phase
    torch.save({"per_head_mean": qns.per_head_mean, "within_group_spread": qns.within_group_spread,
                "layers": qns.layers}, out_dir / "tensors" / "query_norm_stats.pt")
    torch.save({"pairs": sim.pairs, "pair_groups": sim.pair_groups, "layers": sim.layers,
                "query_cosine": sim.query_cosine, "score_corr": sim.score_corr,
                "bos_logit_corr": sim.bos_logit_corr}, out_dir / "tensors" / "within_group_similarity.pt")
    (out_dir / "observations.json").write_text(json.dumps(obs, indent=2))

    figures: Dict[str, Path] = {}
    if make_figures:
        figures["query_norms"] = plot_query_norms(qns, out_dir / "figures" / "analysis1_query_norms.png", show)
        figures["compatibility"] = plot_compatibility(inp, sim, out_dir / "figures" / "analysis2_compatibility.png", group=0, show=show)
        figures["bos"] = plot_bos_competition(bc, out_dir / "figures" / "analysis3_bos_competition.png", show)
        figures["divergence"] = plot_divergence(sim, out_dir / "figures" / "analysis4_divergence.png", show)

    return Phase32Results(inp, head_tbl, qns, sim, bc, obs, figures, csv_path)


def _save(fig, out_path: PathLike, show: bool) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.show() if show else plt.close(fig)
    return out_path
