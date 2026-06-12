At first the original plan was the run giant experiments, collect lots of data, plot graph then analyse later  
But while doing so problems obviously occurs which transformed the research workflow to run tiny experiment, check logs immediately, verify metricsx, scale up later.

This is more effective for a person new to such research experiment like me and is a good way to prepare to the final attention sink project.

The first experiment was observing learning rate behavior in training where I created a training_log.csv to log iteration number, train loss, val loss, grad norm. 

That training session was successfull.  
Dataset: tiny shakespeare  
Device: cpu
Model:
- n_layer = 6
- n_head = 6
- n_emb = 384

Problem 1: The result shows training on CPU was significantly slower than expected. Running the whole 5000 iterations would take multiple hours which is too inefficient.

My decision was to create a smaller "research mode" configuration, reducing the n_emb, batch_size, max_iters, and eval_interval to mitigate computation burden.  
Lesson learned: CPU is not ideal for long-run experiments.  

Problem 2: Loss curve plot turns out to be 2 flat line.

Gradient norm repidly decrease from ~10 to lower than 1 in the first 20 iterations which suggests that the model likely experiences large paramenter updates, then transitions into a more stable optimisation regime.  
But, training and evaluation loss plot appear to be a flat line, I have investigated this and the cause is by a logging problem where I put the logging code outside the iteration block which makes it only log the last value.  
Lesson learned: Always verify logs by checking raw CSV rows and compare them against terminal outputs.
