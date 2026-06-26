# A Mathematical Framework for Transformer Circuits
This is my notes on watching "A Walkthrough of A Mathematical Framework for Transformer Circuits" video on youtube by one of the author Neel Nanda.

## Why does this paper exist?
We take a neural network, and say that this is not uninterpretable.  
The NN, language otherwise, had human-interpretable algorithms, and internal coherence.

So the paper is essentially reverse engineering it and decompile it to a human-understandable algorithm.  
This is super important from an alignment and safety point of view, for the advancement and integration of AI into the world. We need to know what's going on inside rather than just observing inputs and outputs.

This is the idea of Mechanistic Interpretability.

The understanding of residual stream and attention heads is the conceptual foundation for understanding transformer circuits.  
Key claims: Attention heads are fundamentally information movement mechanism.  
Everything else is secondary.

The residual stream store information at each position.  
The attention system then determines:
- where to read information from
- what information to read
- how to write it back into the residual stream

Attention layers are the only mechanism in a transformer that can move information between positions.

## What is an Attention Head?

An attention head consists of two largely separable components:

- Attention pattern: "Which previous positions should I look at?"

  Given a sequence: A B C
  At each destination position, the head computes a probability distribution over itself and previous tokens.

  For example, position C attends to A, B, C with attention score 0.1, 0.7, 0.2 respectively.
  This probability distribution is called the Attention Pattern/Matrix.

  The structure of an Attention Matrix is a lower triangle with all row sum up to 1, because of causal mask in which attention cannot look forward, and the softmax which generates probabilities that must sum up to 1.

- Value Retrieval + Output

  After deciding where to look, the head recieves information from source positions.

  For each source position: Residual stream -> $W_v$ -> Value vector

  The residual stream at each position $x_i$ is projected to $v_i = W_v x_i$

  The result vector is then calculated as $r = \sum a_iv_i$. This is simply the weighted average of value vectors.

The result vector lives in the head space(d_head), which need to be returned to model space(d_model) using $W_o$  
So the output is $W_o r$, then it can be added to the residual stream x <- x + o

Full process:
1. Start with residual stream X with shape (pos, d_model)
2. Value projection: $W_v x$, shape (pos, d_head)
3. Attention matrix: A with shape (destination, source)
4. Result vector $r = \sum a_iv_i$
5. Output projection: $W_o r$, $W_o$ has shape (d_head, d_model)
6. Complete head computation is $A(X W_v)W_o$

## Linearity

Transformers are ridiculously linear.  
Source of linearity:
- Residual Stream: x <- x + o, extremely additive
- OV  circuits: linearly map $W_v$ and $W_o$
- Head outputs: Output = $\sum Head$, additive

This allows decomposition. Because of this linearity, we can break the network down into pieces. Instead of one giant inscrutable function, we can analyse
- Head A contribution
- Head B contribution
- MLP contribution
- Embedding contribution
separately.

This is the foundation of mechanistic interpretability.

So the Attention Head naturally decomposes to:
- QK circuit: decide where to attend
- OV circuit: decide what information to transport

It is more useful to think of the attention head as these 2 circuits rather than QKV independently.

## Relevance to Attention Sink project

THis section is particularly important because attention sinks are not primarily an OBV phenomenon - they are a QK phenomenon.

When investigating attention sink, my main objective study will be the QK and the resulting attention matrix A.

The key research question become:
- why do many heads assigns persistent mass to the BOS/first token?
- which layers and heads exhibit sink behavior?
- is the sink caused by query geometry or key geometry or both?
- How does sink attention interact with residual stream norms?
- does the sink survive after RoPE, NTK,..?

The model suggested to be used is Qwen3-1.7B which use GQA(Grouped Query Attention):
- 28 transformer layers
- 16 Query heads
- 8 Key heads
- 8 Value heads
- Context length: 32K tokens.
