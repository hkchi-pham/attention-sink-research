# StreamingLLM notes
This is my notes reading and watching videos on the paper "Efficient Streaming Language Models with Attention Sink".

--> Solve a problem that exist when you try to run a model beyond the window you trained it on.

Discover an interesting property of any token at position during pretraining(Attention sink)

Attention sink apparently serve to keep the attention scores/softmax distribution stable.

## The problem  
If you have a pretrained model, supposedly, each token will attend to itself and previous tokens in training.  

Sequence: x1, x2, x3, x4.  
x1: attend to itself  
x2: attend to x1 and itself  
x3: attend to x1, x2, and itself  
x4: attend to x1, x2, x3, and itself

So the causual/masked attention matrix will look like:  
    1 2 3 4  
1 *  
2 * *  
3 * * *  
4 * * * *  

But then you may need to run inference on a new token(e.g. x5, x6,...), at some point you're going to reach the limit of your memory.

So one of the solutions to keep running inference is to keep a sliding window, and only consider tokens within the window context. But this require the model to recompute attention for every sequence. This increases the time and memory complexity.

The solution to this is the idea of KV cache:  
Because of the causual attention, token can only attend to the previous tokens, so the computation at the position will stay the same. So we can cache all of those computation so that we only need to compute the new layer every inference.

However, this mechanism break down as we use the sliding window technique.  
Sliding window defines a different computation. The cached KV are still valid for the computation that originally produced them, but they may no longer match the hidden states that a strict sliding-window recomputation would produce. Reusing them is therefore an approximation whose quality depends on how much those hidden states actually change.  
If you want the computation to be strictly true regarding the sliding window context, you would have to recompute every part again.
MUCH SLOWER THAN USING CACHE!!

Some people tried keeping the cache while still using sliding window, as if the 2nd and 3rd,... token can still attend to the first. But the problem is the new token cannot attend to the first token(the paper proposes how this doesn't work).

This hypothesis states how the first token acts as a regularization token where it absorbs all of the extra attention that was not allocate to any other token in the sequence.

Through training, the model has learnt to allocate these extra/not-needed attention to the first token, so all attention scores sum up to 1.  

The paper provide evidence experimenting with different model, comparing:
- Dense Attention
- Window Attention
- Sliding window with recomputation
- Streaming LLM(Their attention sink method)

Plotting the perplexity of the model as input length increases, they show how ther technique stay low perplexity while still using cache so it can go beyond training context length.  
The heat map of attention scores throughout layers also prove the existence of the attention sink(a red line where the position 0 is) -> This is what my experiment is trying to achieve.

## The StreamingLLM Method  
-> keep the 0 token around while maintaining the sliding window.

Generating token 7: [0,1,2,3] [4,5,6] [7]  
Generating token 8: [0,1,2,3] [ ] [5,6,7] [8]  
Generating token 9: [0,1,2,3] [   ] [6,7,8] [9]

Here [0,1,2,3] are the attention sink tokens, [   ] are the evicted tokens, then the next is the sliding window token.  
This allow  the model to keep the benefit of the KV cache while not explosding perplexity.  

The paper also ask "is that always the case?" with attention sink: replace the first token with a "\n" token, and they get the same result. This shows how the behavior of attention sink relies on the position instead of the semantic meanings of the tokens.

Then, the researchers ask: "Can we train this?", like add a null token to the start position of the sequence. The short answer is: YES.

## Interesting notices

The paper states that StreamingLLM focus on positions within the cache rather than those in the original text.  
e.g If the current cache has [0,1,2,3,6,7,8], and in the process of inferencing the 9th token, the positional encoding would be [0,1,2,3,4,5,6,7] rather than [0,1,2,3,6,7,8,9].

At first, it seems that if StreamingLLM can reassign positions inside the KV cache, then Sliding Window Attention should be able to do the same.  
e.g if the current window only contains the original position [8,9,10,11], why not simply relabel them as [0,1,2,3]?

So that a position 0 exists again and can act as an attention sink?

The answer is that the attention sink is not simply "whatever token has position 0". During pretraining, the model repeatedly, sees the first few tokens of a sequence remain present throughout the entire decoding process. Because attention weights must always sum to 1, the model gradually learns to dump excess attention onto these **persistent** early tokens whenever no previous token is particularly relevant. Therefore, the sink is a structural role learned during training, not merely the numeric label.

StreamingLLM preserves this learned behavior by **keeping the original sink tokens** (e.g., positions 0–3) in the cache permanently while sliding only the recent window. It may re-index the cached positions internally because RoPE only depends on relative position differences, so shifting every surviving token by the same offset preserves the attention computation.

In contrast, Sliding Window Attention completely removes the original beginning tokens. Renaming

8 9 10 11

to

0 1 2 3

does not recreate the attention sink, it only tells the model that these later tokens are at the beginning of the sequence, a situation it was never trained to handle. The model has learned to rely on persistent early tokens, not on the integer value of the position itself.

Key idea: RoPE allows positions to be shifted because only relative distances matter, but the attention sink depends on preserving the original beginning tokens that have remained in the cache since the start of generation.

## Beyond StreamingLLM

StreamingLLM reveals that not all important tokens are semantically meaningful.

Two different notions of importance exist:

- Architectural importance: tokens that stabilize the attention mechanism (e.g., attention sink tokens). These tokens may receive attention from many later queries despite carrying little semantic information.
- Semantic importance: tokens whose content is needed later (e.g., variable definitions, names, facts, assumptions).

These are independent concepts. A token can be architecturally important without containing useful semantic information, and vice versa.

StreamingLLM retains

[ Sink Tokens ] + [ Recent Sliding Window ]

and discards all middle tokens.

This cache policy is purely position-based.

Although this preserves the attention mechanism, it assumes that all middle tokens are equally disposable. In practice, important semantic information may exist anywhere in the discarded region.

Therefore, StreamingLLM primarily addresses architectural stability, not long-range semantic memory.

An intuitive improvement would be to retain semantically valuable historical tokens instead of selecting tokens solely by position.

However, the fundamental challenge is: Future usefulness is unknown during autoregressive inference.

A token that has received little or no attention so far may become critical thousands of tokens later.

Different cache policies can estimate token importance using different signals.

| Strategy                         | Advantages                                         | Limitations                                               |
| -------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| Position (StreamingLLM)          | Extremely simple; no extra computation             | Ignores semantic importance                               |
| Accumulated attention            | Measures historical usage; inexpensive to maintain | Favors attention sinks and frequently attended tokens     |
| Exponential Moving Average (EMA) | Emphasizes recent usage                            | May gradually forget historically important tokens        |
| Maximum attention                | Captures strong interactions                       | Sensitive to isolated spikes; may overestimate importance |
| Learned predictor                | Potentially estimates semantic usefulness          | Requires training data and additional computation         |


StreamingLLM intentionally solves a narrow problem: preserve stable attention behaviour while maintaining constant-memory inference.

It does not attempt to solve optimal long-range memory retention. A more general cache policy could combine multiple types of information, for example:

Cache =
    Sink Tokens
  + Recent Window
  + Selected Historical Tokens

where historical tokens are selected using an estimate of future usefulness rather than position alone.

Designing such an importance metric remains an active research problem because semantic usefulness cannot be directly observed during streaming inference.
