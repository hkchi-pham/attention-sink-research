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
The transformer predicts a probability distribution over the vocabulary.  
Initially, non-sensical words like "banana", "table", punctuation can get higher probability like 0.9, 0.87, ... while the correct answer  "Paris" only accounted for ~0.15, which is very wrong.

Training must therefore modify weights so that the probability of "Paris" increases and the probability of other non-sense decrease.

### Loss in transformer

The model prediction for the next token is then compared with the correct target. This produces LOSS.

Small loss --> good prediction.  
Large loss --> bad prediction.

The entire purpose of transformer training is reduce the prediction error across huge amount of text.

A transformer can have millions or billions of parameters. Every paramenter affect loss, so the loss landscape becomes an unimaginably large and high-dimensional space.  
So training means moving parameters toward lower-loss regions.

The transformer needs to know:
- which parameter caused the mistake,
- and how they should change

This information comes from the GRADIENT, which tells us how changing a parameter affect loss.  
Example of parameters can be: token embedding, query matrix, key matrix, value matrix, attention weight, output projection, MLP weights, LayerNorm parameters,...

After loss is computed, backpropagation calculates gradients which then flow backward: output layer --> MLP layer --> attention layer --> embeddings.  
This tells every parameter whether it helped or harmed the predictions.

The core weight update formula is $W \leftarrow W - \eta \nabla_W L$, where we slightly adjust parameter in the direction the reduce loss.

### Trainer

The trainer manages the entire training pipeline. It handles:
- Batching tokens
- forward pass
- loss computation
- backpropagation
- evaluation
- checkpoint saving
- logging
- optimiser calls
It's like an overall training engine.

* Why transformer use batching?

  Transformer datasets are enormous. Training on the entire dataset simultaneously would be impossible. So we split training data in **mini-batches**.

  A batch might contains many sequence of tokens of many length and type. The model process many sequences together on the GPU.

  It is used to improve GPU efficiency, memory usage, and training speed.

  Batching also introduces gradient variability. Different batched learn different language patterns, this slightly changes gradient at each batch which creates noisy/stochastic optimisation. This make the model generalise better as optimiser can't settle in a sharp local minima(e.g. model learn a specific sentence structure).

### Optimiser

The optimiser specifically handles updating transformer parameters. It decides:
- how large updates should be
- how momentum behaves
- how adaptive learning rates work
- how gradients are scaled

1. Stochastic Gradient Descent(SGD)

This is the simplest optimiser.

The general formula is $W \leftarrow W - \eta \nabla_W L$.  
e.g.
W = 2
$\nabla_W L$ = 0.5
$\eta$ = 0.1

==>  $W \leftarrow 2 - 0.1(0.5) = 1.95$

The optimiser would slightly decreases that weight parameter. However, transforer are extremely large, so their optimisation landscape would be more complicated.  
SGC only calculate gradient, take one step, repeat.  
It does not:
- remember past gradient
- adapt learning rate
- stabilise update
- smooth noisy movement
As it only react to current batch gradient, that simplicity becomes a problem for transformer.

A transformer may have to process many different language patterns.  
One batch: code, poetry, dialogue, math, internet slang.  
Another batch: scientific writing, historical text, legal language, emoji.

This produces very different gradient, so optimiser keeps receiving conflicting signals.  
e.g. One batch with internet slang would focus on context, emojis,... While another batch with legal text would focus more on grammar, syntax,... This makes paramenter updates jitter around as direction changes slightly every step.

2. Momentum

  This introduces the idea of accumulated movement direction. Instead of reacting to only the current gradient, the optimiser remembers past updates.

  Momentum helps smooths the transition when gradients fluctuate between batches.

$$
v_t = \beta v_{t-1} + \nabla_W L
$$

$$
W_t = W_{t-1} - \eta v_t
$$

Velocity is basically a moving average of previous updates. So now Momentum SGD considers the overall trend across many repeated steps, instead of reacting to only one batch.

Intuition:
- imagine a repeated pattern: "subjects often relate strongly to nearby verbs."
- across many batched, gradient repeatedly push parameters in a similar direction
- momentum accumulate this repeated signal.
- So the optimiser becomes increasingly confident in that direction.

3. AdaGrad - Adaptive Learning rate

AdaGrad introduces an important idea: different transformer parameters should learn at different speeds.

Some parameter receive huge updates, while others barely change. AdaGrad sclaes update separately.  
Supposed:
- common token embeddings receive many updates
- rare token embedding receive few updates
AdaGrad may reduce learning rates for active parameters and preserve larger updates for under-trained parameters.

When we solve a problem using NN, our input data that we give the NN can be of different types too.  
--> Dense: mostly non-zero  
--> Sparse: mostly zero

So we cannot use the same learning rate to optimise these different feature properly.  
AdaGrad transform the original weight update equation to:

$$
W_t = W_{t-1} -
\frac{\eta}
{\sqrt{\alpha_t}+\epsilon}
\nabla_W L
$$

where, $\alpha_t = \sum(\nabla_W L)^2$

So as the iterations happen $\alpha_t$ will grow hence $\eta$ decrease which decrease the weight slowly then converge.

However, as more the number of iterations increase, $\alpha_t$ becomes super high, meaning the learning rate will become extremely small, so eventually training become too slow.

4. RMSProp - smarter adaptive learning

This improve AdaGrad previous weakness. Instead of remembering all past gradient equally, it emphasises recent gradients.

The main reason why $\alpha_t$ can get very high is that the summation is from 1 to t --> ALL ITERATION

Now the weight update equation becomes:

$$
W_t = W_{t-1} -
\frac{\eta}
{\sqrt{w_{avg}}+\epsilon}
\nabla_W L
$$

where $w_{avg} = \beta v_{t-1} + (1-\beta)(\nabla_W L)^2$

There's no summation in RMSProp, we restrict the growth of $w_{avg}$ as the gradient in multiply by a very small number. So $\eta$ will still decrease, but not as much as in AdaGrad.

5. Adam - The dominant transformer optimiser

Most modern transformer train using Adam or AdamW. It combines:
- Momentum
- RMSProp-style adaptive learning

Momentum equation: $v_t = \beta v_{t-1} + (1-\beta) G_t$  
This is actualy an Exponentially Weighted Moving Average(EWMA). When we expand it recursively the equation become:

$$
v_t =
g_t
+
\beta g_{t-1}
+
\beta^2 g_{t-2}
+
\cdots
$$

Older gradient receive exponentially smaller weights. Thus recent gradient will dominate, distant gradients fade away.  

A large $\beta$, like 0.9, will retain 90% of previous velocity with a longer memory and smoother trajectory, but this will react slower to sudden change.  
Whereas a small $\beta$, like 0.1, will be highly reactive to current gradient and rapidly forget the previous one, but this also creates noisy update and less stable.

EMWA approximately remember $\frac{1}{1-\beta}$ steps.  
Momemtum solve oscillation, slow directional convergence, but NOT per-parameter learning rate, this is still $\eta$.  
So Adam combines momentum and RMSProp.

First moment:
- mean: E[g]
- track by $m_t$

Second moment:
- uncentered variance estimate: $E[g^2]$
- track by $v_t$

So the gradient is $g_t = \nabla_\theta L(\theta_t)$.  
The first moment will estimate $m_t = \beta_1 v_{t-1} + (1-\beta_1) g_t$ which is the momentum, tracking average direction.  
The second moment will estimate $v_t = \beta_2 m_{t-1} + (1-\beta_2)(g_t)^2$, tracking the average squared magnitude of gradient.

Initially $m_0 = 0$ and $v_0 = 0$. This creates a bias problem where initial learning is very slow.

So Adam correct this mathematically with:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
\quad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

The recurrence:  
1. Start with: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$.  
2. Expand to: $m_t = \beta_1^2m_{t-2} + (1-\beta_1)\beta_1 g_{t-1} + (1-\beta_1)g_t$.
3. Repeat until: $m_t = (1-\beta_1)\sum_{i=0}^{t}\beta_1^i g_{t-i}$

The final equation tells us the $E[m_t]$ is a weighted average of past gradients where recent gradients get larger weight and older gradient decay exponentially.

Now assume that the gradients are sampled from some distribution with mean = E[g]  
So $E[g_i]$ = E[g] for all i.  

Expand the formulation with this.  
1. Start with $E[m_t] = (1-\beta_1)\sum_{i=0}^{t}\beta_1^i E[g_{t-i}]$
2. Factor the $E[g_i]$: $E[m_t] = (1-\beta_1)E[g_{t-i}]\sum_{i=0}^{t}\beta_1^i$
3. Here the sum is $\sum_{i=0}^{t}\beta_1^i = 1+ \beta_1 + \beta_1^2 + ... + \beta_1^t$. This is a geometric series. 
4. Using the sum of a geometric series we get $\sum_{i=0}^{t}\beta_1^i = \frac{1-\beta_1^t}{1-\beta_1}$.
5. The term $(1-\beta_1)$ wil cancel out and we're left with $E[m_t] = (1-\beta_1)E[g]$

Even though the true mean is E[g], Adam's moving expectation is $(1-\beta^t)E[g]$ which underestimate the gradient. This is the bias problem.

But now that we discover the culprit to be $(1-\beta^t)$, we can just divide by it.  
So the new $m_t$ and $v_t$ is:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
\quad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

So the final Adam Update rule is:

$$
\theta_t = 
\theta_{t-1} - 
\eta \frac{\hat{m}_{t-1}}{\sqrt(\hat{v_{t-1}}) +\epsilon}
$$

Now Adam simultaneously solve:
- noisy gradient --> momentum averaging
- oscillation --> directional smoothing
- varying parameter sensitivity --> adaptive scaling
- sparse gradient --> parameter-wise learning rates
- AdaGrad decay --> forgetful weighted average
- initialisation bias --> bias correction

However, modern transformer rarely use Adam directly. Because in the full pipeline, L2 regularisation add $\lambda \theta$ to the gradient, but then Adam scale the gradient adaptively which create inconsistent regularisation strength across parameters.

The solution is we decouple weigth decay from gradient update instead of embedding decay into gradient: $\theta \leftarrow \theta - \eta\lambda\theta$ perform those separately.
