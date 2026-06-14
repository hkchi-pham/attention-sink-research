# Experiment 01 - Learning rate

## Goal

Verify that a small nanoGPT model can learn on
Tiny Shakespeare and establish baseline metrics.

## Configuration

n_layer = 2  
n_head = 2  
n_embd = 64

max_iters = 300

## Learning rate = 1e-3 (Ideal)

### Results and Observations

**Training Loss**: 4.20 -> 3.00  
The graph shows a consistent decrease, no sudden spikes, no oscillation, and the model is learning !!! This means the optimiser is successfully finding parameter updates that reduces prediction error.  


**Validation Loss**: 4.19 -> 3.03  
This shows that the model is improving even on data it wasn't trained on, confirming that the learning is effective and the model is not just memorising train data and overfit, but generalisation is actually improving.  

Another important observation is the training curve and the validation curve are almost overlapping, usually the val loss would be greater than the train loss. This perhaps suggests that the model is still early in trainin which make sense as we reduced the training iterations.

**Gradient Norm**:  
Initial ~1.05  
Peak ~1.6  
Final ~0.7

This is interesting as we see a bump, it rises first before it decreases. Now I can see this link to the LR warm up phase. In early training the lr is very small then it slowly increases to max_lr during warm up, and the larger updates often produce larger gradients, so the initial rise is expected.  
But as training progresses loss also decreases, meaning the prediction gets better. This lessens error signal decrease so gradient magnitude will also decrease. This may also links to the cosine decay where learning rate get smaller and smaller so the gradient can converge.

The model learned stable representations and showed no evidence of overfitting within 300 iterations.

## Learning rate = 1e-2

### Results and Observations

**Training Loss**: 4.19 -> 2.48  
The loss increase way more than lr = 1e-3 within 300 iterations, make sense as the model is taking bigger step in optimisation. The gradient in 1e-2 train loss curve als is steeper than the 1e-3 one as the moodel approach convergence quicker. The bit jitter at the end may be due to the fact that the learning rate is too high that it wasn't able to converge smoothly.

**Validation Loss**: 4.19 -> 2.42  
The validation loss decrease even more, showing that the model can learn and didn't overfit/underfit even when the learning rate is high.

**Gradient Norm**  
The gradient norm expectedly fluctuate a lot and didn't follow any decreasing or increasing pattern. Even though it does increase up to ~25th iteration then decrease uniformly to the 100 iterations. But after that, perhaps when approaching convergence point, the model starts to overshoot badly and the gradient update fluctuate dramatically as it oscillates over the narrow valley.

## Learning rate = 1e-4

### Results and Observations  

**Training Loss**: 4.19 -> 3.81  
The decrease shows that the model did learn but slower than when lr = 1e-3. This is expected as the learning rate is decreases. We can see form the graph the the gradient of the train loss curve decrease slower too, with a little bit more subtle jitter. Perhaps with the slower learning the stochastic noise from batch sampling becomes more visible.

**Validation Loss**: 4.19 -> 3.83  
The model also learned during validation too, showing that it didn't overfit when taking smaller step. Similar logic to the training loss, the learning is slower because the lr is decrease form 1e-3 to 1e-4.

**Gradient Norm**:  
Inital ~ 1.04  
Peak ~ 1.60  
Final ~ 1.26

The gradient norm graph also has an initial increase. However, the bump for 1e-3 is only up to the 60 iterations then the gradient starts to decrease. But here, the gradient continue to increase up to ~190 iteration, and the final gradient norm is still higher than the initial norm. Meaning that loss stays high for longer as the model learn slower, so the gradient hasn't been able to converge within 300 iterations.

We can see that with a learning rate of 1e-4, the model is still able to learn even though at a much slower and inefficient rate..

## Learning rate = 1e-5

### Results and Observations

**Training Loss**: 4.20 -> 4.15.  
This is a huge difference to previous run of lr = 1e-3, here the model barely learn at all, it only decreased marginally over 300 iterations. From the graph, we can also see that the values jitter a lot instead of a smoother curve when lr = 1e-3. Because usually, the signal from learning is higher than the noise from batch sampling to the overall curvve moves downward clearly. Whereas when lr = 1e-5, the learning signal becomes too small, so the random variatio form diferent batches becomes comparable to the update size.

**Validation Loss**: 4.19 -> 4.15  
Again, the loss barely changed, the model likely have not learned any thing at all. However, the graph shows a smoother curve than the training loss, this is because validation use loss = estimate_loss() which averages over many batches and remove most of the noise. The trend of very slow learning is shown clearer here.

**Gradient Norm**:  1.05 -> 1.23  
This is interesting as the graph shows an increasing trend instead of a decrease trend. This is because the model isn't moving much. Previously, the gradient norm starts to converge after the warm up stage, but here even after the warm up stage the max learning rate was still too small for the model to learn anything. Loss stays high, hence gradient stays high.

We can see that the learning rate is sufficiently small to maintain stability but too small to achieve efficient optimisation within 300 iterations.

| LR   | Final Val Loss | Convergence Speed | Stability |
| ---- | -------------- | ----------------- | ---------| 
| 1e-5 | 4.15           | -0.00013 loss/iter| High     | 
| 1e-4 | 3.83           | -0.0012 loss/iter | High     |
| 1e-3 | 3.03           | -0.0039 loss/iter | High     | 
| 1e-2 | 2.42           | -0.0059 loss/iter | Low      |



