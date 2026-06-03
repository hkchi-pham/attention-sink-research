## Activation Functions

An activation function in ML is a mathematical function used in neural networks to decide wheter a neuron should be *activated* or not.

It takes the result of a neuron(weighted sum of the features) --> transform it into actual output.

where, z = $w_1x_1 + w_2x_2 + ... + w_nx_n$

Then $\alpha$ = F(z) where F(z) is an activation function which $\alpha$ is the actual information passed to the next layer.

Commonn activation functions:
* ReLu:
  - f(x) = max(0,x)
  - most widely use
* Sigmoid:
  - output values between 0 and 1
  - use in binary classification
  - can suffer from vanishing gradient
* Tanh:
  - output between -1 and 1
  - zero-centered(better than sigmod in some case)
* Softmax:
  - used in multiclass classification
  - convert outputs into probabilities

Overall, activation functions decide how strongly the neuron should respond to input. This helps the network learn complex relationship.

The older(more widely use) is the ReLU
Newer activation is the GeLU

ReLU is extremely simple = max(0,x). It keeps the positive values and remove the negative ones.

For each output x:
- if x > 0, then output = x
- if x <= 0, then output = 0

e.g. input = [-2, -1, 0, 3 , 5] then the output is [0, 0, 0, 3 , 5]

This creates sparsity which helps efficiency.

For x > 0: then gradient = 1, meaning learning continue normally.
For x <= 0: then gradient = 0, meaning neuron stops learning.

This creates a problem called Dying ReLU where neuron stucks outputting 0 forever.

The newer adaptation GeLU is more subtle and probablistic in nature.

F(x) = x . $\phi(x)$, where $\phi(x)$ is a cummulation distribution function of a normal distribution.

Instead of a hard cut off like ReLU, GeLU:
- keep all the value
- scale them base on how large they are where large positive number is kept almost fully, and small negative number is reduced.

<img width="474" height="185" alt="OIP" src="https://github.com/user-attachments/assets/d09a6916-ad86-45c9-b052-c056ca94f5c1" />

In practice, we use an approximation because $\phi(x)$ is expensive.

$F(x) \approx 0.5x(1 + tanh(\frac{\sqrt{2}}{\sqrt{\pi}} (x + 0.044715x^3)))$

GeLU acts like a soft, probabalistic gate instead of blocking all negative, it keeps inputs based on how likely they are useful.

HOWEVER, the real modern winner is the GLU(Gated Linear Unit).
- instead of just transforming the data
- we control(gate) the flow of information

The problem starts with the normal FFN(x).

$FFN(x) = GeLU(W_1x + b_1)W_2 + b_2$

This treats all features equally after activation with no mechanism to selectively control information flow.

$GLU(x) = (xW_c) \cdot \sigma(xW_g)$

- Take input x
- Compute 2 projections into content and gate
- Apply the sigmoid to the gate vector $\sigma(xW_g)$ which scales the value between 0 and 1
- Multiply them together

So, $xW_c$ is the information passed on from each feature and $\sigma(xW_g)$ is how important it is.

The model can now suppress irrelevant features, more flexible the simple activation like ReLU/GeLU.



