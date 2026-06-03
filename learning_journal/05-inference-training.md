## Inference and Training

* Why do model behave differently at training and inference stage?

During training, the model is guided step-by-step using correct answer. But during inference(real use), the model must rely on its own previous predictions.

Lets do a recap:

When you input a sentence like "What is AI?", the model does NOT see text directly. This question is broken into tokens, each token then convert into number ID in the vocabulary, then to the input vectors representing the questions.

Then these input embeddings are inputted into the encoder which holds the attention system where the model explores what each token means and its relationship with other tokens.

The last stage is the decode where, given the input meaning, the model has to generate output sentence step-by-step.

* Training

Teacher forcing: The model is always given the correct previous word when learning from the training dataset.

Input: "What is AI?"  
Target output: "AI is artificial intelligence."

In the decoder:
| input | expected output| 
|-------|-------|
| <strt> | AI |
| <strt>AI | is |
| <strt>Ai is | artificial |
| <strt>Ai is artificial | intelligence |

=> Always given the correct previous token at each epoch.

Training is PARALLEL, the model process alll token at once. It does not generate word-by-word in real time.

At each position:
- model predict next word
- compare with correct word
- compute error(cross-entropy loss)

We then average all error and update weights.  
So during training, the model learns "if previous words are correct, then what comes next?"

* Inference

The first stage is prefill where the model process the prompt.

Then comes the second stage where the model does step-by-step generation.

1. Input = <start> --> predicts "AI"
2. Input = <start>AI --> predicts "is"
3. Input = <start>AI is --> predicts "artificial"
4. Input = <start>AI is artificial --> predicts "intelligence"

Inferecence is SEQUENTIAL.  
- each step depends on the previous step
- cannot parallelise generation.

Here, the model uses its own prediction as input, not the correct one.  
This creates a problems = EXPOSURE BIAS.

The correct path is <start> -> AI -> is -> artificial -> intelligence.

As the model have never seen this sequence before, errors are likely to accumulate.  
e.g. <start> -> AI -> are -> many -> ??? based on the previus sequence it has seen during training.

So during training, we have to apply causual masking carefully so the model cannot "peak" and see the future token which breaks learning cycle. We have to enforce the model to predict the future tokens without seeing them.

Conceptually, a transformer learns by making a decision, measuring how wrong it is, then adjust to reduce error.

1. Forward pass
   - input goes through encoder(embeddings, Attention, FFN,...)
   - target goes to the decoder
   - Model predicts next tokens at each position
2. Compute loss
   - for each position, th emodel output a vector of scores(logits)
   - convert those to probabilities, then compares with correct words.
   - calculate the cross-entropy loss.
3. Backpropagation
   - finds out which part cause the error
   - the loss flows backward thorugh the model, can every parameters get a signal to "increase" or "decrease" something
   - for instance the attention weights, Q/K/V matrices, FFN weights,...

