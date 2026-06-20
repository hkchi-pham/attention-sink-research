This is the report files on the experiments I did in exploring data loading in nanoGPT.

## Visualise batches  
A simple experiment where I print out X and the corresponding Y:
### Result and Observation
I printed the first item in the batch and this is the output:
```
tensor([56, 40, 59, 56, 63,  6,  0, 31, 47, 56,  1, 32, 46, 53, 51, 39])
tensor([40, 59, 56, 63,  6,  0, 31, 47, 56,  1, 32, 46, 53, 51, 39, 57])
X:
rbury,
Sir Thoma

Y:
bury,
Sir Thomas
```

Then to check consistency I also print the 15th and the 86th item:
```
tensor([56, 43,  1, 58, 53,  1, 51, 43, 43, 58,  0, 46, 47, 51,  8,  0])
tensor([43,  1, 58, 53,  1, 51, 43, 43, 58,  0, 46, 47, 51,  8,  0,  0])
X:
re to meet
him.


Y:
e to meet
him.
```
```
tensor([53,  1, 39, 52, 57, 61, 43, 56,  1,  5, 35, 46, 53, 53, 54,  6])
tensor([ 1, 39, 52, 57, 61, 43, 56,  1,  5, 35, 46, 53, 53, 54,  6,  1])
X:
o answer 'Whoop,

Y:
 answer 'Whoop, 
 ```
When I first learn transformer I imagined predictions to be like:  
- input = sentence
- output = next word

But GPT actually learns:  
- position 1 -> next token
- position 2 -> next token
- ...

The prediction target it the train sentence shift by one to the left, so effectively the model is learning what is the next token based on the current tokens.  
For a block size 16, there will be 16 predictions every forward pass

## Randomness check
I want to see the randomness in the batches
### Result and Observation
```
for _ in range(5):
    X,Y = get_batch("train")
    print(X[0][:20])

tensor([ 6,  0, 21, 57,  1, 39,  1, 44, 53, 59, 50,  1, 58, 56, 39, 47])
tensor([42, 63, 12,  0,  0, 15, 27, 30, 21, 27, 24, 13, 26, 33, 31, 10])
tensor([ 1, 61, 53, 59, 50, 42,  1, 39, 57, 57, 39, 63,  6,  1, 54, 56])
tensor([39, 54, 54,  5, 42,  1, 53, 59, 58,  1, 39, 52, 42,  1, 42, 56])
tensor([52, 42,  1, 57, 43, 47, 64, 43,  1, 46, 47, 51, 57, 43, 50, 44])
```
We can clearly see that every batch has a different starting position. Even though if we generate more batch than the block size, the first position will get repeated, this is often but repeating the whole batch is much rarer.  
The repetition is actually necessary, supposed the model sees "to be or not to be" once, have one gradient update, then never again. The model would not learn that pattern well, but if in training the model sees similar examples multiple times, the accuracy will improve
