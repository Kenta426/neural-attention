"""
Batch loader
"""

from load import load_data
import numpy as np
import pickle


class Batcher(object):
    def __init__(self, max_length = 65, data_type = 'dev'):
        # load dev data
        self.data, dictionary = load_data(data_type = data_type, save = False)
        with open('./data/word2id.pkl', 'rb') as f:
            self.word2id = pickle.load(f)
        with open('./data/id2word.pkl', 'rb') as f:
            self.id2word = pickle.load(f)
        # self.word2id, self.id2word = dictionary
        self.vocab_size = len(self.word2id)
        # max sequence length
        self.max_length = max_length

        # build embeddings
        self.p_embedding = np.zeros((len(self.data['targets']), self.max_length), dtype=int)
        self.p_length = np.zeros((len(self.data['targets'])), dtype=int)
        self.h_embedding = np.zeros((len(self.data['targets']), self.max_length), dtype=int)
        self.h_length = np.zeros((len(self.data['targets'])), dtype=int)
        self.t_embedding = np.zeros((len(self.data['targets']), 4), dtype=int)

        self.prepro()

    def prepro(self):

        def convert(w):
            return self.word2id[w] if w in self.word2id.keys() else self.vocab_size

        target_map = {"contradiction":0, "entailment":1, "neutral":2, "-":3}

        for t in ['hypothesis', 'premises', 'targets']:
            if t == 'hypothesis':
                for i, tokens in enumerate(self.data[t]):
                    # insert the end of beginning token
                    self.h_embedding[i] = [self.vocab_size] + list(map(convert, tokens)) + [0] * (self.max_length - len(tokens)-1)
                    self.h_length[i] = 1 + len(self.data[t][i])

            elif t == 'premises':
                for i, tokens in enumerate(self.data[t]):
                    # insert the end of beginning token
                    self.p_embedding[i] = list(map(convert, tokens)) + [0] * (self.max_length - len(tokens))
                    self.p_length[i] = len(self.data[t][i])

            else:
                for i, tokens in enumerate(self.data[t]):
                    idx = target_map[tokens[0]]
                    self.t_embedding[i][idx] = 1

    def next_batch(self):
        pass

Batcher()
