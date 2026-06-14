# Experiment 01 - Learning rate

## Goal

Verify that a small nanoGPT model can learn on
Tiny Shakespeare and establish baseline metrics.

## Configuration

n_layer = 2  
n_head = 2  
n_embd = 64

max_iters = 300

## Learning rate = 1e-3

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

## Learning rate = 1e-5

### Results and Observation

**Training Loss**: 4.20 -> 4.15.  
This is a huge difference to previous run of lr = 1e-3, here the model barely learn at all, it only decreased marginally over 300 iterations. From the graph, we can also see that the values jitter a lot instead of a smoother curve when lr = 1e-3. Because usually, the signal from learning is higher than the noise from batch sampling to the overall curvve moves downward clearly. Whereas when lr = 1e-5, the learning signal becomes too small, so the random variatio form diferent batches becomes comparable to the update size.

**Validation Loss**: 4.19 -> 4.15  
Again, the loss barely changed, the model likely have not learned any thing at all. However, the graph shows a smoother curve than the training loss, this is because validation use loss = estimate_loss() which averages over many batches and remove most of the noise. The trend of very slow learning is shown clearer here.

**Gradient Norm**:  1.05 -> 1.23
This is interesting as the graph shows an increasing trend instead of a decrease trend. This is because the model isn't moving much. Previously, the gradient norm starts to converge after the warm up stage, but here even after the warm up stage the max learning rate was still too small for the model to learn anything. Loss stays high, hence gradient stays high.

We can see that the learning rate is sufficiently small to maintain stability but too small to achieve efficient optimisation within 300 iterations.



