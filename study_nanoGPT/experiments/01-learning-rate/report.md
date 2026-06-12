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

**Convergence speed**  
Initial Val Loss ~ 4.19  
Final Val Loss ~ 3.03  
So the loss reduction is 4.19-3.03 = 1.16, over 300 iterations which makes the average reduction rate 1.16/300 ~ 0.0039 loss per iterations.  

The model learned stable representations and showed no evidence of overfitting within 300 iterations.
