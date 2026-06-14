## Data Loading

We see ```X,Y = get_batch()``` at the beginning of the training loop.  
So what is this function actually doing and why does it atter.  
At first, I thought data loading only plays an insignificant part, but then I realises there's actually a lot to it when I see the detail of the code in nanoGPT.  

I started to ask myself:  
- why random sampling?
- why not split the batch in order?
- why contiguous chunk?
- why shift the target by 1 token?

So this is the notes I took exploring the ```get_batch()``` function. 

The purpose of ```get_batch()``` is:
1. Load a random chunk of text
2. Create input sequence X
3. Create target sequence Y
4. Return a mini batch for training

The output is:
```
X.shape = (batch_size, block_size)
Y.shape = (batch_size, block_size)
```

For example:  
X = "The cat sat"
Y = "cat sat on"

### Loading dataset  
```
if split == "train":
  data = np.memap(...)
else:
  data = np.memap(...)
```

Use ```memap()``` instead of loading the train.bin() entirely to RAM. The memap() behave like a file on disk <-> virtual array.  
Only the portion of required data are loard. This matter because GPT data set can be 10GB, 100GB, 1TB which can't fit into RAM.  

We use tiny-shakespeare dataset to experiment so memap() doesn't make much of a difference, but for long-run large scale GPT it is crucial.

### Random start positions  
```
ix = torch.randint(
  len(data)-blocksize,
  (batch_size,)
  )
```

This generates an array of randome positions. e.g. [1456, 9321, 457, 8000,...]  
Without this randomness, batches would apppear in order that may make the model overfit or learn correlated updates which ruin the purpose of training so that it can generalised to unseen data.  
The random sampling decorrelated batches which improves SGD.

### Construct X
```
x = torch.stack((
      torch.from_numpy(
        data[i:i+block_size]
      )
      for i in ix
    ])
```

Supposed:  
data = "The cat sat  on the mat"  
i = 10  
block_size = 4  
Then:  
X = "cat sat on the"

This creates a context window with the length of  ```block_size```  

But, why contiguous chunks?  
GPT learns next token prediction which requires **order**. If we randomized the tokens then there's no language structure left to learn.

### Construct Y
```
y = torch.stack((
      torch.from_numpy(
        data[i+1:i+1+block_size]
      )
      for i in ix
    ])
```

Notice i --> i+1, the train sentence is shift 1 token to the left. So at every token, the model predicts the next one. 

Here, the model is not learning the sentence meaning directly. It just learns how to predict the correct next toekn given all information given(attention, token embedding, ffn,...).  

Then the last stage is to move the data(x,y) from CPU RAM to GPU VRAM if the device have GPU.  

**Connection to attention sink project**  
Notice how every sequence starts somewhere random. But then position 0 receive more attention than other position??

Understanding how sequence are sampled is the first step toward understanding why certain postion become special during training.
