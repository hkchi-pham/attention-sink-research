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

