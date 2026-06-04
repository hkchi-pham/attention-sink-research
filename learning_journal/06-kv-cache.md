## KV Cache

Cache: stored information reused later, often used in:
- CPU
- Browsers
- Games
- Database

This specific cache is stored in the GPU(VRAM)
### Why do we need KV cache?

Modern transformer language models such as OpenAI's GPT model generate text one token at a time.  
The problem is as the sentence becomes larger, the model has more prvious tokens to look at.
- more computation
- more memory usage
- slower inteference
This is why long-context models are exoensive and memmory-hungry.

The core idea of KV cache is instead of recomputing old information repeatedly, we store it.  
That stored information is called the Key-Value Cache. This remembers the previously computed keys and values, so the model does not need to recompute them again and again.

e.g. During interference, the model genereate "The cat sat.".  
Without KV cache:
1. Generate "The" --> compute Q, K, V for "The".
2. Generate "cat --> compute K, V for "The" again for attention
3. Process continues

This keeps repeating forever which is extremmly inefficient for long sentences where there's lots of previous words.  
The model repeatedly recalculates old tokens K, V even though the old tokens never change. This makes the computation grows roughly $O(n^2)$.

The key insight needed when using KV cache is, one a token is already processed, the key and value NEVER CHANGE!!  
So instead of recomputing them, we store them in memory.

With the KV cache, the complexity is only O(n), because at every step the model only need to compute on pair of K and V.

| Advantage | Disadvantage |
|-----------|--------------|
| Faster generation | More memory usage |
| Avoid repeated computation | Cache grows with sequence length |
| Low latency after prompt | Large context will need large VRAM |

The importance difference between prompt processing and generation in KV cache usage:
1. Prompt processing:
   
   Supposed the prompt is "Tell me about dragons."
   
   At this moment, no cache exist yet and the model has to compute K and V for all prompt tokens.
   --> Slower
3. Autogressive Decoding
   
   Now the model starts generating "Once upon a time ..."
   
   At this point, cache does exist and old K and V values are reused.
