# FlashAttention: Exact Tiled Attention with Online Softmax

## Core Idea

FlashAttention is a **systems optimization** of scaled dot-product attention.

It does **not** change the mathematical definition of attention.

Instead, it reorganizes the order of computation so that the same output is produced while dramatically reducing memory traffic between High Bandwidth Memory (HBM) and on-chip memory.

The mathematical attention remains

$$
Attention(Q,K,V) =
Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

The optimization is entirely in **how** this computation is executed.

---

## Motivation

Vanilla attention computes

$$
QK^T
\rightarrow
\text{Softmax}
\rightarrow
AV.
$$

A straightforward implementation materializes two large intermediate tensors:

* logit
  
$$
S=QK^T
$$

* attention matrix

$$
A=Softmax(S).
$$

These tensors are repeatedly written to and read from HBM even though they are only temporary values.

For long sequences, memory movement dominates runtime rather than arithmetic.

---

## High-Level Strategy

Instead of computing the entire attention matrix first

FlashAttention partitions Q, K, and V into tiles.

Each tile is loaded into fast on-chip shared memory, reused many times, and discarded once it has contributed to the final output.

The computation becomes

```text
Load Q tile
    ↓
For every K/V tile:
    Compute logits
    Update softmax statistics
    Update partial output
Discard tile
```

The full attention matrix is never materialized in HBM.

---

## Why Naive Tiling Does Not Work

Softmax is computed over an entire attention row.

For logits

$$
[x_1,x_2,\ldots,x_n],
$$

the probability of token (i) is

$$
\frac{e^{x_i}}
{\sum_j e^{x_j}}.
$$

If only one tile is available,

the denominator is incomplete.

Therefore computing softmax independently for each tile would produce incorrect probabilities.

The challenge is therefore: Compute the exact softmax without storing the complete attention matrix.

---

## Online Softmax

FlashAttention processes one tile at a time while maintaining sufficient statistics for each query row.

Instead of storing all logits,

it stores only three running quantities.

### 1. Running Maximum

$$
m =
\max(s_1,\ldots,s_k).
$$

Purpose:

* prevents exponential overflow,
* provides the reference point for numerically stable softmax.

---

### 2. Running Normalization Constant

Instead of storing every exponential,

FlashAttention stores

$$
l =
\sum_j e^{s_j-m}.
$$

This is the denominator of the numerically stable softmax.

---

### 3. Running Output Accumulator

The numerator of attention is

$$
N =
\sum_j e^{s_j-m}v_j.
$$

The final attention output is

$$
o =
\frac{N}{l}.
$$

Thus FlashAttention never needs the individual attention weights once they have contributed to the accumulated numerator.

---

## Updating the Running Statistics

Suppose the current running maximum is

$$
m_{old}.
$$

A later tile contains a larger value

$$
m_{new}.
$$

Since

$$
e^{s-m_{new}} =
e^{s-m_{old}}
\cdot
e^{m_{old}-m_{new}},
$$

every previously accumulated exponential can be converted simply by multiplying by

$$
e^{m_{\text{old}}-m_{\text{new}}}.
$$

This scaling factor depends only on the old and new maxima, **not** on the individual logits.

Therefore previous logits never need to be stored.

The normalization constant updates as

$$
l_{\text{new}} =
e^{m_{\text{old}}-m_{\text{new}}}
,l_{\text{old}}
+
\sum_{\text{new tile}}
e^{s-m_{\text{new}}}.
$$

Similarly, the accumulated numerator updates as

$$
N_{\text{new}} =
e^{m_{\text{old}}-m_{\text{new}}}
,N_{\text{old}}
+
\sum_{\text{new tile}}
e^{s-m_{\text{new}}}v.
$$

After all K/V tiles have been processed,

$$
o =
\frac{N}{l}.
$$

This produces **exactly** the same result as vanilla attention.

---

## Why FlashAttention Is Exact

FlashAttention changes

* execution order,
* memory layout,
* scheduling,

but NOT the mathematical computation.

Each attention score is still computed.

Each exponential is still included in the normalization.

Each value vector still contributes to the weighted sum.

The only difference is that intermediate tensors are replaced by running statistics.

The remaining differences are only normal floating-point rounding effects.

---

## Memory Flow

### Vanilla Attention

```text
HBM
 ↓
Read Q,K
 ↓
Compute QKᵀ
 ↓
Write logits S
 ↓
Read logits S
 ↓
Compute Softmax
 ↓
Write attention A
 ↓
Read attention A
Read V
 ↓
Compute AV
 ↓
Write Output
```

Large temporary tensors are repeatedly transferred between HBM and on-chip memory.

---

### FlashAttention

```text
HBM
 ↓
Load Q tile
 ↓
Load K/V tile
 ↓
Shared Memory
 ↓
Compute logits
 ↓
Online softmax update
 ↓
Update output accumulator
 ↓
Repeat for remaining K/V tiles
 ↓
Write final output
```

The attention matrix is never stored in HBM.

---

## Recomputation During Training

During the forward pass,

FlashAttention avoids storing the logits and attention matrix.

During the backward pass, these intermediate quantities are recomputed from Q and K when needed.

This is cheaper than storing and later reading massive intermediate tensors from HBM.

This idea is closely related to activation checkpointing:

* trade additional computation,
* for significantly lower memory traffic.

---

## Numerical Stability

FlashAttention uses the numerically stable softmax

$$
e^{x_i-m},
$$

where

$$
m=\max(x).
$$

Subtracting the maximum:

* prevents overflow
* reduces underflow
* keeps exponentials within a numerically safe range
* preserves the exact softmax because subtracting the same constant from every logit does not change the probability distribution.

The running maximum allows this stabilization to be maintained even while processing attention incrementally.

---

## Computational Characteristics

FlashAttention:

* performs approximately the same number of FLOPs as vanilla attention
* produces mathematically identical output
* dramatically reduces HBM memory traffic
* greatly increases arithmetic intensity
* is substantially faster on modern GPUs because attention is memory-bound rather than compute-bound.

---

## Key Takeaways

* FlashAttention is an implementation optimization, not a new attention mechanism
* The primary optimization is eliminating materialization of the full attention matrix in HBM
* Online softmax enables exact attention to be computed incrementally.
* Three running quantities are sufficient:

  * running maximum,
  * running normalization constant,
  * running output accumulator.
  
* When a larger maximum is encountered, previous accumulations are rescaled by

$$
e^{m_{\text{old}}-m_{\text{new}}}
$$

rather than recomputed.

* Training recomputes intermediate tensors during the backward pass instead of storing them.
* FlashAttention achieves higher performance by reducing memory traffic while preserving identical mathematical attention.
