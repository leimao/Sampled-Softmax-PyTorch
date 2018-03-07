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
├── generated.txt
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

Training performance:

```
| epoch   1 |   200/ 1452 batches | lr 0.10 | ms/batch 215.45 | loss 11.36 | perplexity 85596.94
| epoch   1 |   400/ 1452 batches | lr 0.10 | ms/batch 162.75 | loss  5.75 | perplexity   312.90
| epoch   1 |   600/ 1452 batches | lr 0.10 | ms/batch 128.38 | loss  4.38 | perplexity    79.69
| epoch   1 |   800/ 1452 batches | lr 0.10 | ms/batch 113.95 | loss  4.21 | perplexity    67.08
| epoch   1 |  1000/ 1452 batches | lr 0.10 | ms/batch 107.85 | loss  4.11 | perplexity    61.21
| epoch   1 |  1200/ 1452 batches | lr 0.10 | ms/batch 106.52 | loss  4.02 | perplexity    55.87
| epoch   1 |  1400/ 1452 batches | lr 0.10 | ms/batch 103.48 | loss  3.99 | perplexity    54.20
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 201.94s | valid loss  5.33 | valid perplexity   205.78
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 1452 batches | lr 0.10 | ms/batch 105.83 | loss  3.93 | perplexity    51.16
| epoch   2 |   400/ 1452 batches | lr 0.10 | ms/batch 106.76 | loss  3.84 | perplexity    46.71
| epoch   2 |   600/ 1452 batches | lr 0.10 | ms/batch 104.03 | loss  3.69 | perplexity    40.09
| epoch   2 |   800/ 1452 batches | lr 0.10 | ms/batch 103.06 | loss  3.69 | perplexity    40.11
| epoch   2 |  1000/ 1452 batches | lr 0.10 | ms/batch 104.36 | loss  3.69 | perplexity    40.08
| epoch   2 |  1200/ 1452 batches | lr 0.10 | ms/batch 105.92 | loss  3.67 | perplexity    39.34
| epoch   2 |  1400/ 1452 batches | lr 0.10 | ms/batch 117.72 | loss  3.69 | perplexity    40.05
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 163.68s | valid loss  5.07 | valid perplexity   159.08
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 1452 batches | lr 0.10 | ms/batch 105.11 | loss  3.67 | perplexity    39.18
| epoch   3 |   400/ 1452 batches | lr 0.10 | ms/batch 104.26 | loss  3.61 | perplexity    37.00
| epoch   3 |   600/ 1452 batches | lr 0.10 | ms/batch 104.11 | loss  3.51 | perplexity    33.37
| epoch   3 |   800/ 1452 batches | lr 0.10 | ms/batch 103.97 | loss  3.53 | perplexity    34.17
| epoch   3 |  1000/ 1452 batches | lr 0.10 | ms/batch 102.62 | loss  3.55 | perplexity    34.67
| epoch   3 |  1200/ 1452 batches | lr 0.10 | ms/batch 101.57 | loss  3.53 | perplexity    34.05
| epoch   3 |  1400/ 1452 batches | lr 0.10 | ms/batch 102.73 | loss  3.56 | perplexity    35.11
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 158.75s | valid loss  5.02 | valid perplexity   150.88
-----------------------------------------------------------------------------------------
| epoch   4 |   200/ 1452 batches | lr 0.10 | ms/batch 103.41 | loss  3.54 | perplexity    34.42
| epoch   4 |   400/ 1452 batches | lr 0.10 | ms/batch 102.23 | loss  3.51 | perplexity    33.29
| epoch   4 |   600/ 1452 batches | lr 0.10 | ms/batch 102.46 | loss  3.41 | perplexity    30.38
| epoch   4 |   800/ 1452 batches | lr 0.10 | ms/batch 102.75 | loss  3.44 | perplexity    31.09
| epoch   4 |  1000/ 1452 batches | lr 0.10 | ms/batch 102.27 | loss  3.46 | perplexity    31.67
| epoch   4 |  1200/ 1452 batches | lr 0.10 | ms/batch 102.10 | loss  3.44 | perplexity    31.31
| epoch   4 |  1400/ 1452 batches | lr 0.10 | ms/batch 102.45 | loss  3.48 | perplexity    32.52
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 157.85s | valid loss  4.94 | valid perplexity   139.76
-----------------------------------------------------------------------------------------
| epoch   5 |   200/ 1452 batches | lr 0.10 | ms/batch 105.09 | loss  3.46 | perplexity    31.77
| epoch   5 |   400/ 1452 batches | lr 0.10 | ms/batch 104.63 | loss  3.43 | perplexity    30.96
| epoch   5 |   600/ 1452 batches | lr 0.10 | ms/batch 104.42 | loss  3.35 | perplexity    28.42
| epoch   5 |   800/ 1452 batches | lr 0.10 | ms/batch 105.70 | loss  3.37 | perplexity    29.05
| epoch   5 |  1000/ 1452 batches | lr 0.10 | ms/batch 103.80 | loss  3.39 | perplexity    29.59
| epoch   5 |  1200/ 1452 batches | lr 0.10 | ms/batch 105.68 | loss  3.39 | perplexity    29.67
| epoch   5 |  1400/ 1452 batches | lr 0.10 | ms/batch 104.77 | loss  3.42 | perplexity    30.62
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 160.68s | valid loss  4.93 | valid perplexity   137.73
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.92 | test perplexity   136.84
=========================================================================================
```

To generate a sample essay:
```bash
python generate.py
```

Generated sample essay:

```
districts for years to import april as dozen ray ivy news director rebound from aetna keith william rica robinson an
estimated $ N billion because of importance or faberge and the nation 's markets so administrative bread high salon will
suffer and indeed rising admission on race taxes he added but efforts to improve exposure credit need packages and are
focused on abortions to congress canada europe say beyond the century that could execution their capital defeat liquidity by selling
industry could emerge from other attracting in europe but over five years if it was clear not that could be
dangerous in nuclear control though mr. <unk> would surveyed the political interested match it says in more affairs than many
carolina first claims would be completed by dec. ag compared with an raising covert viable task that california researchers and
send the compromise even not submit with them can still not fit mr. ridley 's counsel in the list of
coups <eos> but prudential-bache liability funding by federal funding are behind business federal brown income talks <eos> sometimes applications will
returns backlogs to bureau but cease daily other leval and related medical programs makes decisions on requirements movement and dr.
western creditors <eos> both universities overseas expired materials still based on cargo program for brief time not necessarily responsibility is
likely to keep imports or more efficient air division stores at st. tackle westinghouse parts with double that as many
as banks trade and electronic fined rules <eos> leverage by requiring cancer dating N hours other investigation of carter projects
with a optimism about 's slight primary price <eos> and according to an international accountants banks oppose union media boss
cnw industries ltd. morgan ltd. statement rejected a judge but no guarantee by widespread conservatives ' meeting to this education
contend they see that any consequences of concern <eos> this is n't treated as an early efforts mainly in large
fastest-growing southern adjacent especially westridge inc. has ruled that what 's request <eos> edward spokesman said sea technology did n't
deny what it might explain we are not not disappointed that what was really anything wrote they thought it simply
said friday grain company traditionally agreed to withdraw from its N stake in delaware executives at alex brothers n.y <eos>
mccaw declined month concedes we 're along with refineries we mature nov. N <eos> without circuit development authorized offering three
pediatric alternative pension package in N <eos> bellsouth 's l.p. expanded from $ N million or c$ N a share
a year earlier <eos> firstsouth inc. retains its majority of public securities to its headquarters in value dragged on oct.
N <eos> and an unrelated promotion of pitney small-town last year is a best seller paid while some investors are
held or professionals <eos> this trade between health and acquisitions are provided for been awful zero for partnership that we
're necessary to do high-risk tapes both sectors of heavy aid as part of program and unilever control groups <eos>
unless males and stay on specified rates will pay loans per record price due and $ N hangs less but
also do n't provide assurance <eos> it 's even actively treated according to some concerns over costa cases cover assuming
that any alternative unsuccessfully has served just as $ N billion in loans which he got more than $ N
billion in gramm-rudman emergency value profit <eos> some analysts say that lawmakers are in fiscal N made some tax increase
in injury stemming from europe until N N later <eos> congress went measure process to seek at least vote on
federal funds service data with an estimated $ N billion with the authors less bond at a bit wall street
while rep. kennedy is chairman of sept. r. cohen and nonetheless holders who do n't want <unk> <eos> but like
gulf programs that left trying to fm topics on routes not rico likely generation such plans because of agreements surrounding
crews clearing earthquake cholesterol and sanctions to diplomatic panama that meanwhile their leader modify cooperatives enforcement <eos> both prosecutors do
n't want laurence police to represent pinkerton 's table in law and nature of health budget hide plan and dramatic
cia sources health committees and the u.s. conducting space over transportation how threatened an phony standing extensive that a pending
compromise for pregnancy and press <unk> through italy place <eos> or systems however were suspension on shopping cutting their entire
coup nicaragua 's heading assistance <eos> moreover it was given amid possible sale abortion forces already are used in regulated
despite federal agencies in their hope to preserve their tax restructure because of government rates will require soon grossly <unk>
costs onto neighbor government cooled <eos> sen. whitten striking r. witness james r. ind. ariz. joint form of raiders from
chairman and chief executive officer and chief executive officer said according to gop rep. dale reflect robert coleman who began
rep. r. d. boren <unk> having learned that the mandatory have been succeeded by sen. cranston morning to pass on
speculation that war democrats do n't have disclose that debt or assassinations would be manufacture but often just between china
but its dubious agents will likely support off an reduction in parliament legislators from time to speak in speculative robinson
republicans or democracy behind who policy maintenance agents convert a concrete reunification or other class of <eos> assistance tax memories
landscape high wage bill murder military assistance of these tv president appeared one of this month learned <eos> statements nearly
fewer funds are often controls for tainted exports <eos> pilots ' leaders in new york endorsed china 's hud experience
inquiry sharply in N of its spending bill indicating secretary carlos nixon told him that mr. bush would have there
are an issue <eos> mr. coleman head italy dan jamie meanwhile did n't vote after relevant trading on dusty dissent
said evans d. r. calif. attorney for abortion secretary runkel opponents argued that it has been reached for corp <eos>
```


## Note

Weight tying for sampled softmax has not been implemented yet.