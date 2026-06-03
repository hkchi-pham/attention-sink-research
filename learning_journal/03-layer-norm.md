Layer Normalization (LayerNorm) stabilizes training by keeping activations at a consistent scale across layers. This helps gradients flow more reliably through deep networks, reducing the risk of exploding or vanishing values and making training faster and more stable.
## Pre-norm vs Post-norm

# Post-norm
The residual is added BEFORE the LayerNorm:
```text
x
│
├───────────────┐
│               │
▼               │
Multi-Head      │
Attention       │
│               │
▼               │
+ ◄─────────────┘
│
▼
LayerNorm
│
▼
output
```
Formula: $y = \text{LayerNorm}(x + \text{MHA}(x))$

And the FFN block is $y = \text{LayerNorm}(x + \text{FFN}(x))$

# Pre-norm
The residual is added AFTER the LayerNorm:
```text
x
│
├─────────────────────┐
│                     │
▼                     │
LayerNorm             │
│                     │
▼                     │
Multi-Head Attention  │
│                     │
▼                     │
+ ◄───────────────────┘
│
▼
output
```
Formula: $y = x + \text{MHA}(\text{LayerNorm}(x))$

FFN block: $y = x + \text{FFN}(\text{LayerNorm}(x))$

---------------------------------------------------------------------------------------------------

* PreNorm is better as it creates:
  - more stable training: in PostNorm, values can explode for become unstable during training.
  - better gradient flow: In PostNorm, output = LayerNorm(x+transform(x))

So gradient must go through LayerNorm after addition which can distort/shrink gradient.

But in PreNorm, output  = x + transform(LayerNorm(x)). Gradient goes directly through connection which helps it stays stable through many layers.
This remove the need for warm-up in PostNorm to work well.

* But why is a PostNorm bad?
  
  Think of the residual flow as a direct connection between input to output, acting as a highway for information and gradient to bypass the layer's computation and be added directly to its output.
  
  If you insert the LayerNorm in between, you break the identity path, the gradient value struggle to pass through.

A new idea of a Double Norm also emerge --> normalise before AND after MHA, FFN,... to create more stability for complex model.

----------------------------------------------------------------------------------------------------
# LayerNorm vs RMSNorm

* LayerNorm(the original):
  - subtract by $\mu$
  - divide by $\sigma$
  - then scale(weight) and shift(bias)

* RMSNorm(modern):
  - Only divide by the magnitude
  - no bias term

RMSNorm is better as it is faster and just as good as LayerNorm but with fewer parameters and operations.
- no bias term to store
- no mean calculation

Operations such as matrix multiplication(tensor contraction) takes up most of the computer's flopp(~99%), while normalisation only took ~0.2%.

So sometime the question of why do we even optimise normalisation comes up, but the importatn insight is, in speed, the flop is not the only factor contributing. But DATA MOVEMENT is also significant.

In normalisation, there're not much flop coverage but there's a lot of head movement --> 25% of runtime.
So using RMSNorm still matter due to the importance of memory movement.


