# Understanding Qwen for Attnetion Sink research 

## Why Qwen?
The goal of this project is to investigate attention sink behavior in multilingual decoder-only transformers, with a focus on English and Vietnamese.

Qwen is chosen because it provides:
- Open weights and reproducible research access
- Strong multilingual capability
- Modern transformer architecture
- Long-context support
- Accessible attention weights for interpretability experiments.

Unlike GPT-2, which is primarily an English model from 2019, Qwen incorporates more recent architectural improvements and is designed for long-context inference. This makes it a better platform for studying attention sinks in realistic modern LLMs.

## Qwen Family
Qwen is not a single model but a family of models:

Qwen (general language model)
Qwen-Chat
CodeQwen
MathQwen
Qwen-VL (vision-language)
Qwen-Audio

The project will primarily use a general-purpose Qwen model because the objective is to study attention mechanisms rather than specialised capabilities.

## Architecture Overview
Qwen follows the decoder-only transformer architecture.

Pipeline:
Input Tokens  
↓  
Tokenizer (BPE)  
↓  
Token Embeddings  
↓  
RoPE Positional Encoding  
↓  
Multi-Head Self-Attention  
↓  
Feed Forward Network (SwiGLU)  
↓  
Output Projection  
↓  
Next Token Prediction

## Untied Embeddings
Qwen uses untied input embeddings and output projection matrices.  
Many language models share the same matrix for converting tokens into embeddings and converting hidden states back into vocabulary logits

This is called weight tying. Qwen instead learns separate matrices.

Advantages:
- More expressive representations.
- Greater flexibility during generation.
- Slightly improved performance.

This design choice is unlikely to directly affect attention sink formation.

## Positional Encoding: RoPE
### Why Positional Encoding Exists
Self-attention alone of permutation invariant.  
Without positional information:  
"I love cats"  
and  
"Cats love I"  
appear identical.  
Position information must therefore be injected into the model.

### Rotary Position Embedding(RoPE)
Instead of adding learned position vectors, RoPE rotates Query and Key vectors according to token position.  

Conceptually:
- Q(position 1)
- Q(position 2)
become rotated versions of the same vector

The same applies to Key vectors. Attention scores therefore naturally encode relative position.

RoPE can be understood as many sinusoidal clocks operating simultaneously.  
Different dimensions rotate at different frequencies:
- Low-frequency dimensions rotate slowly
- High-frequency dimensions rotate rapidly

High-frequency components primarily encode local relationships.  
Low-frequency components encode longer-range relationships.

This interpretation becomes important when extending context lengths beyond training.

### Context length
Context length is the maximum number of tokens visible to the model at once.

Example: Training Context Length = 2048, means each token can attend to at most 2048 previous tokens during training.

Longer contexts create two major challenges:
- Positional extrapolation
- Attention dilution

Both are directly relevant to attention sink research.

### Positional Extrapolation Problem
Suppose Qwen is trained on positions **0 → 2047**, but at inference time receives position **7000**

RoPE must generate rotations corresponding to positions the model never encountered during training.

This creates an out-of-distribution positional encoding problem.

Observed effects include:
- unstable attention patterns
- degraded retrieval
- lower accuracy
- increased hallucinations

### NTK-aware interpolation
A simple solution is linear position scaling:

p' = p / s

where:  
s = new_context_length / training_context_length

This compresses positions back into the training range.

However, all frequencies are compressed equally. This distorts local positional relationships and changes attention behaviour.

Not all RoPE frequencies are equally important.  
High-frequency dimensions:
- encode nearby relationships
- support local reasoning

Low-frequency dimensions:
- encode long-range relationships
- support distant retrieval

NTK-aware interpolation modifies RoPE frequencies in a way that preserves local structure while stretching long-range positional information.

The goal is to maintain the original attention kernel as closely as possible.

In practice:
- nearby token relationships remain stable
- long-range context can be extended
- performance degrades more gracefully

This enables Qwen to operate beyond its training context length without retraining.

## Attention Dilution
Longer contexts introduce another challenge.  
Attention uses softmax:

Attention(Q,K)=softmax(QKᵀ)

As context length increases:
- more keys compete for probability mass
- attention becomes more diffuse
- relevant tokens receive less attention

This phenomenon is called **attention dilution**.

Long-context degradation is therefore not only a positional problem.  
It is also an optimisation problem inside the attention mechanism itself.

### LogN Scaling
Qwen introduces LogN scaling to mitigate attention dilution.

Attention logits are rescaled:

score' = score × log(n)/log(n_train)

where:  
n = current context length  
n_train = training context length

Longer contexts create more competition inside softmax.

LogN scaling increases attention sharpness.

Effects:
- stronger focus on important tokens
- reduced attention dilution
- improved long-context retrieval
- better context extrapolation

LogN scaling can therefore be viewed as a mechanism that counteracts the tendency of attention to become overly distributed as sequence length grows.

## Window Attention
Not every layer requires access to the full context.

Empirical observations suggest:  
Lower layers:
- rely heavily on local information
- are more sensitive to context extension

Higher layers:
- perform more global reasoning

Qwen therefore applies window attention in selected layers. Instead of attending to all previous tokens, some layers only attend within a local window.

Benefits:
- lower memory usage
- reduced computation
- better long-context stability

## Multi-Head Attention
Qwen uses standard multi-head self-attention.  
Each attention head learns different patterns.

Examples commonly observed in transformer interpretability studies:
- previous-token heads
- induction heads
- retrieval heads
- sink heads

Understanding these specialised heads becomes important during attention sink experiments.  
This topic will be explored after establishing baseline sink behaviour.

## Flash Attention
Standard attention requires O(n²) memory.

Flash Attention computes attention in memory-efficient blocks.

Benefits:
- reduced memory consumption
- faster training
- faster inference

Importantly: FlashAttention does not change the mathematical attention mechanism. It is an implementation optimisation rather than an architectural change.

## Tokenisation
Qwen uses Byte Pair Encoding (BPE).  
Vocabulary size: ~152K tokens.

BPE constructs tokens from frequently occurring subword units.

Advantages:
- efficient multilingual support
- handles rare words
- handles code and mixed-language text

### Relevance to Vietnamese
Vietnamese tokenisation differs substantially from English.

Examples:

English:
"artificial intelligence"

Vietnamese:
"trí tuệ nhân tạo"

The tokenizer may segment these phrases differently.

Potential consequences:
- different token counts
- different positional distributions
- different attention patterns

This raises an important research question: Does attention sink behaviour vary across languages partly because of tokenisation differences?

## Training
Training Objective:
- Next-token prediction.

The model learns: P(next token | previous tokens)

Training data consists primarily of English and Chinese, supplemented by other languages.  
Data quality is improved through:
- rule-based filtering
- model-based scoring
- selective upsampling of valuable sources

Upsampling means certain datasets are sampled more frequently than their raw size would suggest.  
This helps improve performance on important domains such as code or high-quality text.

## Connection to Attention Sink Research
Attention sinks were first characterised in StreamingLLM (Xiao et al., 2023).

A sink is a token position that receives disproportionate attention across many heads and layers.  
The first token often acts as such a sink.

Key observations:
- Sink behaviour strengthens with context length.
- Removing sink tokens dramatically harms performance.
- Preserving sink tokens enables stable streaming inference.

### Why Qwen is Interesting for Sink Research
Several Qwen design choices may interact with sink formation:

1. RoPE

Attention sinks are fundamentally positional phenomena. RoPE changes how position is represented compared to learned embeddings.

Question: Do RoPE-based models develop different sink behaviour?

2. Long Context Extension

NTK-aware interpolation and LogN scaling change attention behaviour at long sequence lengths.

Question: Do these mechanisms strengthen or weaken attention sinks?

3. Multilingual Tokenisation

Vietnamese and English have different token distributions.

Question: Do attention sinks behave similarly across languages?

4. Window Attention

Some layers see only local context.

Question: Does limiting context alter sink emergence in lower layers?
