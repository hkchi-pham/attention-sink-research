# Attention Computation and Systems Foundations

## Mathematical Attention

Scaled dot-product attention computes

$$
{Attention}(Q, K, V) =
{Softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

where:

* **Query (Q):** Represents what the current token is looking for.
* **Key (K):** Represents what information each token provides for matching.
* **Value (V):** Represents the information passed to future computations.

The attention score between two tokens is the dot product between their query and key vectors. Larger dot products indicate stronger similarity, and the softmax converts these scores into a probability distribution over all tokens.

The final output is a weighted combination of the value vectors using these probabilities.

### Why are Q, K, and V different?

The model separates:

* **Address space:** Queries and keys determine *where* to retrieve information.
* **Payload space:** Values determine *what* information is retrieved.

If (Q=K=V), the same representation would simultaneously determine similarity and transmitted information, reducing the model's flexibility.

---

## Why divide by $\sqrt{d_k}$?

Assume the entries of Q and K are approximately independent with variance 1.

The dot product is

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i.
$$

Since the variance of independent sums adds,

$$
{Var}(q\cdot k)\approx d_k.
$$

Therefore the standard deviation grows like

$$
\sqrt{d_k}.
$$

Dividing by

$$
\sqrt{d_k}
$$

keeps the variance approximately constant.

Without scaling:

* logits become increasingly large
* softmax becomes highly peaked
* gradients vanish because softmax saturates.

Dividing by d_k would over-scale the logits, making attention nearly uniform and reducing the model's ability to distinguish relevant tokens.

The Gaussian assumption is only a mathematical approximation. The important insight is that **the variance of the dot product grows approximately linearly with the vector dimension**, not that the vectors are exactly normally distributed.

---

## Complexity of Attention

Let

* sequence length = (n)
* head dimension = (d)

### Computing QK^T

$$
(n\times d)(d\times n)
\rightarrow
n\times n
$$

Each output entry is a dot product of length d

Complexity:

$$
O(n^2 d).
$$

---

### Softmax

Softmax is applied independently to each row.

Each row requires:

* exponentiation
* summation
* normalization

Overall complexity:

$$
O(n^2).
$$

---

### Computing AV

$$
(n\times n)(n\times d)
\rightarrow
n\times d
$$

Complexity:

$$
O(n^2 d).
$$

---

### Overall complexity

Computation:

$$
O(n^2 d).
$$

Memory:

$$
O(n^2).
$$

The computation scales with both sequence length and head dimension, while the largest intermediate tensor(the attention matrix)depends only on sequence length.

---

## Why the Attention Matrix is the Bottleneck

For

* n=4096
* d=128

Approximate tensor sizes:

| Tensor           | Elements     |
| ---------------- | ------------ |
| Q                | 0.5 million  |
| K                | 0.5 million  |
| V                | 0.5 million  |
| Attention matrix | 16.8 million |

The attention matrix is over 30× larger than each of Q, K, or V.

Doubling the context length:

* Q, K, V grow linearly.
* The attention matrix grows quadratically.

Therefore, long-context attention is primarily limited by the attention matrix.

---

## GPU Memory Hierarchy

Modern GPUs have multiple memory levels with different speed and capacity trade-offs.

| Memory               | Location | Relative Speed               | Capacity   |
| -------------------- | -------- | ---------------------------- | ---------- |
| Registers            | On-chip  | Fastest                      | Tiny       |
| Shared Memory (SRAM) | On-chip  | Very fast                    | Small      |
| L2 Cache             | On-chip  | Fast                         | Medium     |
| HBM                  | Off-chip | Slowest in the GPU hierarchy | Very large |

HBM is not objectively slow. It is only slow relative to on-chip memory.

- **Registers** store **thread-private variables** that are only accessed by a single thread.
  - **Example:** During matrix multiplication, each thread accumulates one output element in a register:
    $c_{ij} \leftarrow c_{ij} + a_{ik} b_{kj} $

    Here, the partial sum $c_{ij}$ is kept in a register until the computation finishes.

- **Shared memory** stores **data reused by many threads within the same thread block**.
  - **Example:** In tiled matrix multiplication or FlashAttention, a tile of the Q or K matrix is first loaded into shared memory:
  $Q_{\text{tile}},\; K_{\text{tile}} \in \text{Shared Memory}.$

    All threads in the block reuse these tiles instead of repeatedly reading them from global memory.

- **HBM (High Bandwidth Memory)** stores **large tensors** that do not fit on-chip.
  - **Example:** Before computation begins, the full attention inputs reside in HBM: the Q, K, V.
    
    During execution, tiles of Q, K, and V are copied from HBM into shared memory, processed, and the output is written back to HBM.

---

## Why Vanilla Attention is Slow

The mathematical computation is

$$
QK^T
\rightarrow
\text{Softmax}
\rightarrow
AV.
$$

A straightforward implementation performs:

1. Read Q and K from HBM.
2. Compute logits S=QK^T
3. Write S to HBM.
4. Read S
5. Compute $A=Softmax(S)$.
6. Write A to HBM.
7. Read A
8. Read V
9. Compute output O=AV
10. Write O

The tensors:

* logits S
* attention matrix A

are only intermediate results. They exist solely to produce the final output but are repeatedly written to and read from HBM.

The primary bottleneck is therefore **memory traffic**, not arithmetic.

Modern GPUs are typically **memory-bound**, meaning computation is often faster than memory can supply data.

---

## Compute-bound vs Memory-bound

A workload is:

* **Compute-bound** when arithmetic operations dominate runtime.
* **Memory-bound** when moving data dominates runtime.

Vanilla attention is primarily memory-bound because repeatedly transferring the large attention matrix between HBM and on-chip memory is more expensive than the matrix multiplications themselves.

---

## Arithmetic Intensity

Arithmetic intensity is

$$
\frac{Computation}{Memory Traffic}.
$$

Increasing arithmetic intensity means performing more computations for each byte loaded from memory.

FlashAttention achieves higher arithmetic intensity by maximizing reuse of data while it remains in shared memory.

---

## Core Insight Behind FlashAttention

FlashAttention does **not** change the mathematical definition of attention.

Instead, it reorganizes the computation to reduce HBM accesses.

Rather than computing:

1. all logits
2. then all softmax
3. then all outputs

FlashAttention computes attention tile-by-tile, keeping small blocks of Q, K, and V in shared memory long enough to complete many computations before writing the final output.

The result is exactly the same attention output while dramatically reducing memory traffic.

---

## Numerical Precision

Softmax is numerically sensitive because exponentials can overflow or underflow.

### Overflow

Large positive logits can produce

$$
e^x=\infty,
$$

leading to undefined expressions such as

$$
\frac{\infty}{\infty}.
$$

---

### Underflow

Very negative logits can produce

$$
e^x\approx0,
$$

causing loss of precision or, in extreme cases, a denominator of zero.

---

### Numerically Stable Softmax

Subtracting the maximum logit

$$
m=\max_i x_i
$$

gives

$$
\frac{e^{x_i-m}}
{\sum_j e^{x_j-m}}.
$$

This transformation leaves the softmax unchanged because adding or subtracting the same constant from every logit does not alter the probability distribution.

Benefits:

* prevents overflow
* greatly reduces underflow
* improves numerical precision
* ensures the largest exponential equals 1 while all others lie in ((0,1]).

---

## Mixed Precision

Modern LLMs often use BF16 or FP16 for efficiency.

Critical reductions, such as:

* maximum computation
* normalization constants
* accumulated sums

are frequently performed in FP32 to maintain numerical stability while preserving the speed and memory advantages of lower precision arithmetic.

