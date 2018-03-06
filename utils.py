import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import data
import math


def word_count(corpus):
    counter = [0] * len(corpus.dictionary.idx2word)
    for i in corpus.train:
        counter[i] += 1
    for i in corpus.valid:
        counter[i] += 1
    for i in corpus.test:
        counter[i] += 1
    return np.array(counter).astype(int)

def word_freq_ordered(corpus):
    # Given a word_freq_rank, we could find the word_idx of that word in the corpus
    counter = word_count(corpus = corpus)
    # idx_order: freq from large to small (from left to right)
    idx_order = np.argsort(-counter)
    return idx_order.astype(int)

def word_rank_dictionary(corpus):
    # Given a word_idx, we could find the frequency rank (0-N, the smaller the rank, the higher frequency the word) of that word in the corpus
    idx_order = word_freq_ordered(corpus = corpus)
    # Reverse
    rank_dictionary = np.zeros(len(idx_order))
    for rank, word_idx in enumerate(idx_order):
        rank_dictionary[word_idx] = rank
    return rank_dictionary.astype(int)



class Rand_Idxed_Corpus(object):
    # Corpus using word rank as index
    def __init__(self, corpus, word_rank):
        self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
        self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
        self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
        self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)

    def convert_tokens(self, tokens, word_rank):
        rank_tokens = torch.LongTensor(len(tokens))
        for i in range(len(tokens)):
            #print(word_rank[tokens[i]])

            rank_tokens[i] = int(word_rank[tokens[i]])
        return rank_tokens

    def convert_dictionary(self, dictionary, word_rank):
        rank_dictionary = data.Dictionary()
        rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
        for idx, word in enumerate(dictionary.idx2word):
            #print(word_rank)
            #print(rank)
            rank = word_rank[idx]
            rank_dictionary.idx2word[rank] = word
            if word not in rank_dictionary.word2idx:
                rank_dictionary.word2idx[word] = rank
        return rank_dictionary



class Word2VecEncoder(nn.Module):

    def __init__(self, ntoken, ninp, dropout):
        super(Word2VecEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        #print(self.encoder.weight[0][0])
        emb = self.encoder(input)
        emb = self.drop(emb)
        return emb

class LinearDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
        return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))


class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq



class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            in_, out_ = self.params.weight.size()
            stdv = math.sqrt(3. / (in_ + out_))
            self.params.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids))
        true_freq = Variable(torch.FloatTensor(true_freq))
        sample_freq = Variable(torch.FloatTensor(sample_freq))

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels, :]
        true_bias = self.params.bias[labels]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids, :]
        sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long())
        return logits, new_targets

    def full(self, inputs):
        return self.params(inputs)

