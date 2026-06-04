# Trainer and Optimiser

### How model actually learn

--> Trainers, Optimisers, Gradients, and Weight Updates

These notes explain how a neural network improves itself during training.  
The goal is not just to memoriese formulas, but to understand:
- what the model is trying to do
- why gradient matter
- how optimiser change weight
- why batching exist
- how algorithm like SGD, Momentum, RMSPRop, AdaGrad, and Adam actually behave

A neural network starts with random weights. At initialisation:
- attention matrices are random
- embeddings are random
- MLP layers are random
- output predictions are almost meaningless

For example, if we prompt "The capital of France is ", an untrained model would predict stuff like "banana", "computer", random punctuations,...  
Because it had not learned the language pattern yet

Training gradually changes the weights so the model learns:
- grammar
- sentence structure
- relationship between words
- token prediction behavior

Most decoder-only(like GPT) model are trained using **Next-token prediction**.  
The model sees previous tokens and tries to predict the next token

For example, the input is "The capital of France is ..." and target output is "Paris".  
The trand=sformer predicts a probability distribution over the vocabulary.  
Initially, non-sensical words like "banana", "table", punctuation can get higher probability like 0.9, 0.87, ... while the correct answer  "Paris" only accounted for ~0.15, which is very wrong.

Training must therefore modify weights so that the probability of "Paris" increases and the probability of other non-sense decrease.

### Loss in transformer

The model prediction is compared against the next token. This produces LOSS.

Small loss --> good prediction.  
Large loss --> bad prediction.

The entire purpose of transformer training is reduce the prediction error across huge amount of text.

A transformer can have millions or billions of parameters. Every paramenter affect loss, so the loss landscape becomes an unimaginably large and high-dimensional space.  
So training means moving parameters towrd lower-loss regions.

The transformer needs to know:
- which parametet caused the mistake,
- and how they should change

This information comes from the GRADIENT, which tells us how changing a parameter affect loss.  
Example of parameters can be: token embedding, query matrix, key matrix, value matrix, attention weight, output projection, MLP weights, LayerNorm parameters,...

After loss is computed, backpropagation calculates gradients then flow backward through the output layer --> MLP layer --> attention layer --> embeddings.  
Thsi tells every parameter whether it helped or harmed the predictions.





