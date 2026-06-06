The entire training loop is basically: get batch --> forward pass --> compute loss --> backward pass --> gradient clipping --> AdamW update --> Zero gradient --> next batch.

### Stage 1 - Get data
```code
X, Y = get_batch('train')
```
where X is the inpute token adn Y is the target token.  
e.g. X = ["The", "cat", "sat"] and Y = ["cat", "sat", on"]

### Stage 2 - Learning rate

At the start of every iteration you could see 
```code
lr = get_lr(iter_num)
```
Then
```code
for param_group in optimiser.param_group:
  param_group['lr'] = lr
```
This is where warm up and cosine decay happen. At first I did not know how or why we have to do warm up learning rate and what cosine decay is because I only learnt the basic knowledge of what learning rate is, but not yet how it will look like in implementation.  
So I read my notes again and notice that intially, the parameter's values are very chaotic, everything is at a random initialisation.  
This makes me realise that is we immediately use a fixed large lr right at the start, updates are too large and this can destroy useful structure and learn non-sense structure before the model has even started learning.

So there has to be a warm up phase:
```code
if it < warmup_iters:
  return lr * (it+1/warmup_iters+1)
```
This means, if we have learning rate = 6e+4 by default, the lr will slowly rise from 0 to the learning rate during warm up phase, so the model can take small step when the parameters are a bit chaotic and begin to learn more once the the model has a little more understanding of the language and gradients are meaningful.

Then,  
```code
if it > decay_iter:
  return min_lr
```
This is the minimum learning rate when training reaches the iteration where gradient is converging, small step isrequired so they don't overshoot. 

In between those steps, we use cosine decay down to the min_lr when we're in the decay iteration range.  
```code
decay_ratio = (it- warmup_iters)/(lr_decay_iters - warmup_iters) #determine how much lr will decay in that iteration
assert 0 <= decay_ratio <= 1
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
return min_lr + coeff*(learning_rate - min_lr)
```
The cosine decay gives a smooth convergence to the min_lr from the max_lr. At first I did not know what this is, but after some research I found out that it's a way to smoothly decrease the learning rate so the model can take smaller step when reaching convergence point --> reach target loss quicker, higher final accuracy, avoid oscillation. Other than cosine decay there are also step decay, exponential decay, ... 

<img width="474" height="278" alt="OIP" src="https://github.com/user-attachments/assets/0abcea7d-fdfa-4bd1-9780-46b061c32a34" />  

See how this create a curve for learning rate to decay smoothly.

### Stage 3 - Forward pass

In model.py, there's the function forward(...) where the model compute token_emb, pos_emb, then the attention layer,...  
Then in MLP stage, tokens get pass through Linear --> GeLU --> Linear.  
Finally we got to
```code
logits = self.lm_head(x)
```
This project n_embs back to vocab size. For every token position, this means the probability of every word in the vocabulary to be predicted next.  
Then we compute loss by compare the logits with the targets, using cross-entropy.

e.g.  
Model predict: "cat" = 0.7, "dog" = 0.2, "bird" = 0.1 with the target = "cat" $\rightarrow$ relatively small loss.  
Model predict: "banana" = 0.77, "table" = "0.12", "cat" = 0.11 $\rightarrow$ huge loss.

So in train.py, we have
```code
with ctx:
  logits, loss = model(X, Y)
  loss = loss / gradient_accumulation steps.
```
The division by gradient_accumulation_steps scale the loss to account for gradient accumulation across batches.

### Stage 4 - Backward pass

```code
scaler.scale(loss).backward()
```
We can ignore the scaler for now, the logic is basically loss.backward().  
Here, PyTorch traverses the computational graph backward.  
For every paramenters, this means that parameter.grad got filled with a value. It is the partial differentiation of loss with respect to that parameter, this tells the model if the model harm or benefit the prediction.

One important observation is that the parameter HAVE NOT CHANGED. loss.backward() only produce the gradients.

### Stage 5 - Gradient Accumulation

Motice the code says
```code
for micro_step in range(gradient_accumulation_step):
```

Supposeed gradient_accumulation_step = 8, then each forward, backward step got repeat 8 times before optimiser.step() is called. This serves the purpose of simulating a larger batch size.   
For example we only have or GPU can only fit 32 examples, but we want 256. Then we repeate the forward, backward step 8 times for all 32 examples $\rightarrow$ accumulate gradients equivilent to those of 256 examples.  
Hence we have to divide loss by this when computing to have the actual loss value.

### Stage 6 - Gradient Clipping

This is also a new technique I learn when exploring the codes.  
So this part,
```code
torch.nn.utils.clips_grad_norm_(model.parameters(), grad_clip)
```
This is essentiall for my future experimental target.

Gradient clipping is basically to prevent the exploding gradient problem by clipping the gradient vector to be within a certain threshold(e.g. -1 < $g_x$ < 1).  
And **gradient clipping by norm** clip the gradient THEN also preserve the proportion of the values by normalising them so gradient still keep its direction. Though this may create vanishing gradient problem if the difference in gradient values are very large and some gradients are normalised to a very small value.

### Stage 7 - AdamW update

```code
scaler.step(optimizer)
```
This is where weights actually change.

PyTorch internally does:  
1. First moment: $m_t = \beta_1 m_{t-1} + (1-\beta_1)G_t$
2. Second moment: $v_t = \beta_2 v_{t-1} + (1-\beta_2)(G_t)^2$
3. Update: $\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt(\hat{v}_t) + \epsilon}$
4. Weight decay: $\theta \leftarrow \theta - \eta\lambda\theta$

Now what get decayed and what doesn't? I learn that some weight actually don't get decay during training for its characteristic.  
Typically, parameters with dimension >= 2 will get decay. Like linear weights, embedding weights, attention matrices, MLP matrices,...  
And parameters with dimension < 2 does not get decay. Like the bias vector or LayerNorm weights(which directly control normalisation, shrinking them toward 0 would hurt optimisation).  

### Stage 8 - Zero Gradient

The final stage is:
```code
optimiser.zero_grad(set_to_none = True)
```
This clears the parameter.grad after every step so old gradient won't add to new gradient so the model learn correctly.
