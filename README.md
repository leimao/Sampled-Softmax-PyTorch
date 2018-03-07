# Sampled_Softmax_PyTorch

Lei Mao

University of Chicago

## Introduction

Sampled softmax is a softmax alternative to the full softmax used in language modeling when the corpus is large. Google TensorFlow has a version of [sampled softmax](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) which could be easily employed by the users. In contrast, Facebook PyTorch does not provide any softmax alternatives at all. 

Ryan Spring has implemented a [sampled softmax](https://github.com/rdspring1/PyTorch_GBW_LM) using PyTorch but his sampling approach was based on C++ codes. Compling his C++ codes sometimes raises problems and running his C++ backended sampled softmax sometimes reports error in some machines. Based on his code, I implemented a similar sampled softmax but the sampling approach was coded in Python using Numpy. I found that at least for the [Zipf distribution](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.zipf.html) (also called Log-Uniform distribution) used for sampled softmax, the sampling is quite efficient using Numpy.

## Dependencies

Python 3.5

PyTorch 0.3

Numpy

## Files

The sampled softmax class, sampling functions, and other helper classes and functions are all coded in ``utils.py`` file. I also modifed Facebook [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) for the implementation test.

```
.
├── data
│   └── ptb
│       ├── test.txt
│       ├── train.txt
│       └── valid.txt
├── data.py
├── generate.py
├── LICENSE.md
├── main.py
├── model.py
├── README.md
└── utils.py
```


## Usage

I added several options to the Word-level language modeling RNN api. But some of the options, such as ``tied``, might not be valid any more due to the modification in the code. I will fix these in the future if I get time.

```
optional arguments:
  -h, --help             show this help message and exit
  --data DATA            location of the data corpus
  --model MODEL          type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE        size of word embeddings
  --nhid NHID            number of hidden units per layer
  --nlayers NLAYERS      number of layers
  --lr LR                initial learning rate
  --clip CLIP            gradient clipping
  --epochs EPOCHS        upper epoch limit
  --batch-size N         batch size
  --bptt BPTT            sequence length
  --dropout DROPOUT      dropout applied to layers (0 = no dropout)
  --decay DECAY          learning rate decay per epoch
  --tied                 tie the word embedding and softmax weights
  --seed SEED            random seed
  --log-interval N       report interval
  --save SAVE            path to save the final model
  --softmax_nsampled N   number of random sample generated for sampled softmax
```


To train a language model using sampled softmax:

```bash
python main.py --softmax_nsampled 200 --model GRU --lr 0.1 --nhid 200 --bptt 10 --batch_size 64
```

```
| epoch   1 |   200/ 1452 batches | lr 0.10 | ms/batch 192.83 | loss  5.59 | perplexity   268.58
| epoch   1 |   400/ 1452 batches | lr 0.10 | ms/batch 134.98 | loss  4.35 | perplexity    77.87
| epoch   1 |   600/ 1452 batches | lr 0.10 | ms/batch 114.73 | loss  4.12 | perplexity    61.40
| epoch   1 |   800/ 1452 batches | lr 0.10 | ms/batch 108.20 | loss  4.05 | perplexity    57.38
| epoch   1 |  1000/ 1452 batches | lr 0.10 | ms/batch 103.62 | loss  3.98 | perplexity    53.60
| epoch   1 |  1200/ 1452 batches | lr 0.10 | ms/batch 101.85 | loss  3.93 | perplexity    50.79
| epoch   1 |  1400/ 1452 batches | lr 0.10 | ms/batch 100.03 | loss  3.91 | perplexity    49.73
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 185.15s | valid loss  5.19 | valid perplexity   179.34
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 1452 batches | lr 0.10 | ms/batch 100.25 | loss  3.79 | perplexity    44.29
| epoch   2 |   400/ 1452 batches | lr 0.10 | ms/batch 100.06 | loss  3.69 | perplexity    40.14
| epoch   2 |   600/ 1452 batches | lr 0.10 | ms/batch 100.33 | loss  3.60 | perplexity    36.68
| epoch   2 |   800/ 1452 batches | lr 0.10 | ms/batch 99.79 | loss  3.62 | perplexity    37.50
| epoch   2 |  1000/ 1452 batches | lr 0.10 | ms/batch 100.83 | loss  3.62 | perplexity    37.34
| epoch   2 |  1200/ 1452 batches | lr 0.10 | ms/batch 98.79 | loss  3.61 | perplexity    37.01
| epoch   2 |  1400/ 1452 batches | lr 0.10 | ms/batch 100.37 | loss  3.63 | perplexity    37.85
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 153.78s | valid loss  5.01 | valid perplexity   150.21
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 1452 batches | lr 0.10 | ms/batch 99.92 | loss  3.58 | perplexity    35.72
| epoch   3 |   400/ 1452 batches | lr 0.10 | ms/batch 99.84 | loss  3.52 | perplexity    33.68
| epoch   3 |   600/ 1452 batches | lr 0.10 | ms/batch 99.78 | loss  3.44 | perplexity    31.23
| epoch   3 |   800/ 1452 batches | lr 0.10 | ms/batch 99.57 | loss  3.47 | perplexity    32.10
| epoch   3 |  1000/ 1452 batches | lr 0.10 | ms/batch 99.57 | loss  3.48 | perplexity    32.54
| epoch   3 |  1200/ 1452 batches | lr 0.10 | ms/batch 99.44 | loss  3.47 | perplexity    32.20
| epoch   3 |  1400/ 1452 batches | lr 0.10 | ms/batch 99.15 | loss  3.50 | perplexity    33.14
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 153.14s | valid loss  4.96 | valid perplexity   142.02
-----------------------------------------------------------------------------------------
| epoch   4 |   200/ 1452 batches | lr 0.10 | ms/batch 100.22 | loss  3.46 | perplexity    31.91
| epoch   4 |   400/ 1452 batches | lr 0.10 | ms/batch 99.94 | loss  3.42 | perplexity    30.48
| epoch   4 |   600/ 1452 batches | lr 0.10 | ms/batch 99.38 | loss  3.35 | perplexity    28.51
| epoch   4 |   800/ 1452 batches | lr 0.10 | ms/batch 99.66 | loss  3.38 | perplexity    29.23
| epoch   4 |  1000/ 1452 batches | lr 0.10 | ms/batch 99.07 | loss  3.39 | perplexity    29.80
| epoch   4 |  1200/ 1452 batches | lr 0.10 | ms/batch 100.02 | loss  3.39 | perplexity    29.74
| epoch   4 |  1400/ 1452 batches | lr 0.10 | ms/batch 99.25 | loss  3.43 | perplexity    30.82
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 153.19s | valid loss  4.92 | valid perplexity   137.37
-----------------------------------------------------------------------------------------
| epoch   5 |   200/ 1452 batches | lr 0.10 | ms/batch 99.88 | loss  3.39 | perplexity    29.77
| epoch   5 |   400/ 1452 batches | lr 0.10 | ms/batch 99.49 | loss  3.36 | perplexity    28.67
| epoch   5 |   600/ 1452 batches | lr 0.10 | ms/batch 99.46 | loss  3.28 | perplexity    26.67
| epoch   5 |   800/ 1452 batches | lr 0.10 | ms/batch 98.75 | loss  3.32 | perplexity    27.54
| epoch   5 |  1000/ 1452 batches | lr 0.10 | ms/batch 99.31 | loss  3.33 | perplexity    28.07
| epoch   5 |  1200/ 1452 batches | lr 0.10 | ms/batch 99.97 | loss  3.33 | perplexity    28.08
| epoch   5 |  1400/ 1452 batches | lr 0.10 | ms/batch 99.84 | loss  3.38 | perplexity    29.30
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 153.08s | valid loss  4.90 | valid perplexity   134.35
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.90 | test perplexity   133.63
=========================================================================================
```

To generate a sample essay:
```bash
python generate.py
```

######
Show demo essay here.
######


## Note

Weight tying for sampled softmax has not been implemented yet.