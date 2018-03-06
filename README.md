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
├── model.pt
├── model.py
├── README.md
└── utils.py
```


## Usage

I added several options to the Word-level language modeling RNN api. But some of the options, such as ``tied``, might not be valid any more due to the modification in the code. I will fix these in the future if I get time.

```bash
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
python main.py --nhid 150 --lr 0.1 --bptt 10
```


######
Show training log here.
######



To generate a sample essay:
```bash
python generate.py
```


## Note

Weight tying for sampled softmax has not been implemented yet.