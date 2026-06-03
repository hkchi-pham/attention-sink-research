## Self-attention

The main idea is each word looks at all other words and decides "which words are important to me?"

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

This is useful because wach head learns different patterns of language

