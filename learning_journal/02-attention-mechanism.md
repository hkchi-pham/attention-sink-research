## Self-attention

**Why attention exist in the first place?**

In ML, we often work with sequences of tokens:
- word in a sentence
- image batches
- audio segment

Each token has its own meaning, but a token does NOT exist alone --> its meaning depends on other token.

e.g. "bank" can mean "river bank", "financial bank", and model can only understand this with context.

A token embedding is a vector that stores the value of the token.
But this is not enough, we need a way for tokens to communicate with each other

This leads to the main idea of attention: each word looks at all other words and decides "which words are important to me?"

Self-attention allows the model to relate the words to each other.
The process is:
1. Create Q, K, V
   These 3 matrices are transformed from the input matrix
   - Query(Q): represents "what kind of information does this word need?"
     
     e.g. word = "bank", then the query may ask "is this money?","is this about river?"
   - Key(K): represents "what kind of information do i offer?"
     
     e.g. word = "river", the key might encode "i am related to nature"
   - Value(V): contains the actualy information to pass to others

   A word compares its Query with all other words' Key to decide how much of their Value it should take.
   
2. Compare Q and K
   
$$
QK^\top
$$

  Each word compares itself with all others, producing a *score* that encode how relevant word i is to word j.
  
  Then we have to normalise these scores:

$$
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

  --> which convert them to probabilities with each row sum to 1.
  
  This is how much attention each word gives to others
  
3. Combines with values

$$ 
\text{Attention}(Q,K,V) =
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
V
$$

  Final representation weighted combination of other words.
  The attention matrix captures not only the meaning or the position in the sentence, but also each word interaction with other words.
  
## Multihead Attention

The problem with single head attention is it only focus on one type of relationship between the tokens.

Solution: Multiple heads
- split the embeddings into smaller parted in different aspect
- run attention multiple times in parallel

e.g split an embedding into 4 head
- head 1: grammar
- head 2: meaning
- head 3: emotion
- head 4: syntax

<img width="809" height="655" alt="3acbb7d3-5f57-4413-888b-ce88c4bc7e7f" src="https://github.com/user-attachments/assets/cde68132-8d69-499e-810b-49a5e70d81b9" />

This is useful because each head learns different patterns of language

## Linear Attention

There's a step in standard attention where we apply softmax row by row:
- convert values into probabilities
- ensures all values are positive and sum up to 1

Now each row is a **probability distribution** of attention.
Next, we multiply attention weights with value vectors.

A BIG PROBLEM with such standard attention is its COMPUTATIONAL ISSUE!!!

If there's n tokens, the attention compute a nxn matrix.
- time complexity: $O(n^2)$
- memory complexity: $O(n^2)$

For long-sequence, this is very slow, memory expensive --> limit scalability.

This brings to an important idea of **generalising attention**.
Initially, attention uses exponential function inside softmax. But instead of exponential, we can use any similarity function.

So now attention = compute similarity between Q and K.

The key idea of linear attention is factorising similarity.

$$
\text{sim}(Q,K) =
\rho(Q)^\top \rho(K)
$$

This is similar to the kernel trick.
Where the original attention $(QK^\top)V$ requires building a whole matrix.
Linear attention rearrange to  $Q(K^\top V)$.

We then apply causual attention to this where a token can only see past tokens, NOT future tokens. So now instead of summing over all tokens, only sum up to current position.
This creates a structure similar to RNN: at each step, we update using previous state + current position.

**Why standard attention is not linear?**

At first glance, attention looks like $(QK^\top)V$.
This looks linear because matrix multiplication is linear and no obvious "weird" function.
But the real formula is 

$$ 
\text{Attention}(Q,K,V) =
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
V
$$

The softmax actually changes everything. It is not just scaling the values to a probability distribution, it is a non-linear transformation.
- take all scores in a row
- applies exponential $\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}$
- then normalise

Here, every token competes with every other tokens. If one score increases slightly, its exponential changes A LOT.
So as output depends on all input together, you cannot separate tokens independently.
If you change one entry in $(QK^\top)V$ all attention weights change, not just one.
So in linear attention, we rewrite attention as:

$$
\frac{\exp(QK^\top)}{\sum \exp(QK^\top)}V
\;\longrightarrow\;
\frac{\text{sim}(Q,K)}{\sum \text{sim}(Q,K)}V
$$

So now we link this to the trick in linear attention: $\text{sim}(Q,K) = \rho(Q)^\top \rho(K)$.

This is important because it separates similarity into 2 independent parts of Q and K. So the output is factorisable into components that depend on only Q or K.

## Slidinng window attention

* Why normal attention doesn't scale?

In a standard transformer, attention is computed using: 

$$ 
\text{Attention}(Q,K,V) =
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
V
$$

Here, every token compares itself withh all other tokens. This produces a full matrix of interactions.

Problem is:
- time complexity = $O(n^2)$
- memory complexity = $O(n^2)$

So the core idea of sliding window attention is, instead of each token attends to ALL otkens, we change to each token only attends to nearby tokens.

e.g. if window size = 2, the token at i attends to i-2, i-1, i, i+1, i+2

So instead of having nxn comparison, we only have nxw comparison where w = window size.

However the obvious trade-off is the attention now cannot see far away.

Token i cannot directly see token i+100 for instance, this may lead to losing hidden pattern in language.

But here's the trick, deeper layers expand the view.

- Layer 1: Token sees nearby token, like the token at i sees the token at i-2, i-1, i, i+1, i+2
- Layer 2: These tokens already saw their neighbour, so token i+1 already has information about token i, passing them to token i+3,i+4,... as well.

So as new layers are built, the information spread.

A more advanced variant of this would be the **dilated sliding window attention**.

This zooms in on a problem of normal sliding window: have strong local understanding, but slow expansion of context.

So the core idea of dilated sliding window attention is: instead of attending to every neighbour, we skip tokens with a fixed step size(dilation).

e.g. window size = 2 and dilation = 2 means token at i attends to tokens at i-4, i-2, i, i+2, i+4.

This keeps the same number of connection, but more spread out so the context coverage expand quicker than normal sliding.

Using dilation still have the trade-off of losing local details as nearby tokens got skipped, and such sparse information will have less precise presentation.

The best practical solution is to have some layer with normal sliding window and some layer with dilation also.

This problem can also be addressed by what is called Global + Sliding Attention.

The problem is even though information spread, some tokens cannot directly see each other which may lead to loss of important informations.

So global + sliding window is a hybrid system that contain both local attention and global attention(special tokens).

Global token is a special token in the sequence that can see the entire sequence, and seen by the entire sequence.

e.g. CLS token, QA question token, manually selected tokens,...

So the normal tokens can attend to their local tokens + global tokens where the global token acts like a central hub storing information about all token and spread them to all other tokens.
This enables long range communication between layers.
