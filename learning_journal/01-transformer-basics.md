# Transformer

## Why do transformer exist?

There are problems with the older model RNN.
They process a sequence one token at a time. At each step, the omdel takes the current input x + the previous hidden state(memory).
Then it outputs a prediction y and a new hidden state.
However, this creates some problems:
1. slow computation
   - RNN process tokens sequentially(like a loop)
   - This means you cnanot process all words at one, the longer the sentence, the slower the training
2. vanishing/exploding gradient
   - When training, the RNN model updates weights using gradient
   - how ever if the gradients are very small, then they shrink further
   - in contrast, if gradients very super large, they will "explode"
3. cannot remember long-range depencies
   - each word lose influence over time
   - e.g. "The cat that was sitting on the mat suddenly jumped"
   - the word "cat" is important for "jumped", but RNN may forget it because it's too far back

The mechanism of transformer solves all 3 problems:
- process everythign ar once
- direct connections between all words
- shorter comptation path

**Transformer structure:**
1. Encoder: process input sentence
2. Decoder: generate output sentences

## Input Representation

1. Tokenisation
   Splitting sentence into tokens(words or subwords.
   e.g. "Your cat is lovely" --> ["Your", "cat", "is", "lovely"]
2. convert to input ID
   Each word is mapped to a number from a vocabulary
   e.g. "cat" = 6500
4. Embeddings
   Each words become a vector
   e.g. the input matrix of sentence "Your cat is lovely" may look like:
   
$$
\begin{bmatrix}
1 & 2 & \cdots & 0 \\
6 & -4 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
3 & 0 & \cdots & -1
\end{bmatrix}
$$

  Where each row is a word. These numbers represents the meaning of the word and they learned DURING training.

## Positional Encoding
Problems occurs as transformer processes all words at one, so it does not know the order.

A solution is we add a postitional encoding by adding a positional vector to the embedding of each word.
Then the input matrix turns to:

$$
I =
\begin{bmatrix}
1 & 2 & \cdots & 0 \\
6 & -4 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
3 & 0 & \cdots & -1
\end{bmatrix}
+
\begin{bmatrix}
0.1 & -2.3 & \cdots & 0 \\
0.04 & 5 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
3 & 0.5 & \cdots & -7
\end{bmatrix}
$$

* How is this calculated ?
  There are many techniques of calculating positional encoding vectors but using sine and cosine is one of the simplest design:
  Words in the same position = same encoding across all sentences. This is fixed because it represents the positon index, not the word itself.

  The same positional encoding is reused, but added to different word embeddings depends on where the words appear.
  Therefore, the same word can have a different representation if it's placed in different position.
  
$$
PE(pos, 2i) =
\sin\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

$$
PE(pos, 2i+1) =
\cos\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

The overall encoder input is token eb=mbeddings + positional encoding.
   
   
