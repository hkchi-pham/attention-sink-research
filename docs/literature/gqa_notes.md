# Grouped Query Attention (GQA)

It's the middle ground between MHA and MQA.
e.g.  
16 Q heads
8 KV heads

Visualise:  
Q1 ─┐  
Q2 ─┘ → KV1

Q3 ─┐  
Q4 ─┘ → KV2

Q5 ─┐  
Q6 ─┘ → KV3  
...

## Motivation

LLMs spend a significant portion of inference time performing attention.

The main computational bottleneck during autoregressive generation is not computing Q, K, and V, it is storing and repeatedly reading the KV cache.

During generation:
- Query are only computed for the newest token.
- Keys and Values from every previous token must be stored.
- Every new token attends to all cached K/V vectors.

The KV cache grows linearly with sequence length.

For modern LLMs with long contexts (32k–128k tokens), this cache becomes one of the largest consumers of GPU memory and memory bandwidth.

## General formulation

Suppose
- $H_q$: number of query heads
- $H_{kv}$: number of KV heads

Then the group size is Hq/Hkv.

For Qwen3-1.7B:
- 16 Q heads
- 8 KV heads

The group size would be 16/8 = 2. Each KV projection serves exactly 2 query heads.

## Computation

Supposed:
- Query heads: $Q_0, Q_1$

Shared:
- $K_0$ head
- $V_0$ head

Query computation: 
- $Q_0 = XW_Q^{(0)}$
- $Q_1 = XW_Q^{(1)}$

Keys: $K = XW_K$  

Values: $V = VW_K$

The Attention score is calculated as:
- $S_0=\frac{Q_0K^\top}{\sqrt{d_{\text{head}}}}$
- $S_1=\frac{Q_1K^\top}{\sqrt{d_{\text{head}}}}$

which form the attention pattern by:
- $A_0=\{softmax}(S_0)$
- $A_1=\{softmax}(S_1)$

The result vectors is the attention multiply by the shared Values: 
- $R_0=A_0V$
- $R_1=A_1V$

And each Query will also has its own output projection:
- $O_0=R_0W_O^{(0)}$
- $O_1=R_1W_O^{(1)}$

Then for one transformer layer, the residual stream update is:

$$
X' =
X
+
\sum_{i=1}^{H_q} O_i
$$

## Computation saving

So the KV cache went from:
- 16 Q heads + 16 KV heads for MHA
- to 16 Q heads + 8 KV heads for GQA

Reduced memory by 50%.  
Reducing KV heads reduces
- cache size
- cache reads
- GPU memory traffic

Inference becomes much faster.

## Mechanistic Interpretability Perspective

Recall, attention consists of:
- QK circuit
- OV circuit

With GQA, within one KV group:
- The K side of the QK circuit is shared, and the V and O side (OV circuit) is also shared.

That means 2 query heads in the same group differ only in their query projection and their output projection. They compute different attention patterns because Q differs, but they compare against the same Keys and retrieve from the same Value vectors.

So if the experiment result shows 2 heads sharing identical K and V exhibit different attention patterns, the difference must originate from their Query projections.

## Advantage and Disadvantage
### Advantage

- Nearly the quality of MHA
- Much smaller KV cache
- Faster autoregressive inference
- Lower memory bandwidth requirements
- Better scalability to long contexts
- Simpler deployment on GPU

### Disadvantage

- Reduced flexibility because Keys and Values are shared within groups.
- Heads within the same group cannot learn completely independent Key or Value representations.
- Some fine-grained specialization may be lost compared with full MHA.
