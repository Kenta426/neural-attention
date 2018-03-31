"""
Batch loader
"""

from load import load_data
import numpy as np
import pickle
import model
import os


class Batcher(object):
    def __init__(self, max_length=model.MAX_LENGTH, data_type='train', batch_size=model.BATCH):
        # check if the model is already saved
        save = not os.path.exists('./data/word_matrix.npy')
        # load dev data
        self.data, dictionary = load_data(data_type=data_type, save=save)
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
        self.t_embedding = np.zeros((len(self.data['targets'])), dtype=int)

        # batch size
        self.batch_size = batch_size
        self.pointer = 0

        self.prepro()
        self.build_batch()

    def prepro(self):

        def convert(w):
            return self.word2id[w] if w in self.word2id.keys() else self.vocab_size

        target_map = {"contradiction":0, "entailment":1, "neutral":2}

        for t in ['hypothesis', 'premises', 'targets']:
            if t == 'hypothesis':
                for i, tokens in enumerate(self.data[t]):
                    # insert the end of beginning token
                    self.h_embedding[i] = list(map(convert, tokens)) + [0] * (self.max_length - len(tokens))
                    self.h_length[i] = len(self.data[t][i])

            elif t == 'premises':
                for i, tokens in enumerate(self.data[t]):
                    # insert the end of beginning token
                    self.p_embedding[i] = list(map(convert, tokens)) + [0] * (self.max_length - len(tokens))
                    self.p_length[i] = len(self.data[t][i])

            else:
                for i, tokens in enumerate(self.data[t]):
                    idx = target_map[tokens[0]]
                    self.t_embedding[i] = idx

    def build_batch(self):
        idx = np.random.permutation(len(self.data['targets']))
        # shuffle
        self.p_embedding = self.p_embedding[idx]
        self.p_length = self.p_length[idx]
        self.h_embedding = self.h_embedding[idx]
        self.h_length = self.h_length[idx]
        self.t_embedding = self.t_embedding[idx]
        # number of batches
        self.n_batches = int((self.t_embedding.shape[0] / self.batch_size))
        # truncate dataset
        self.p_embedding = self.p_embedding[:self.n_batches*self.batch_size, :]
        self.p_length = self.p_length[:self.n_batches*self.batch_size]
        self.h_embedding = self.h_embedding[:self.n_batches*self.batch_size, :]
        self.h_length = self.h_length[:self.n_batches*self.batch_size]
        self.t_embedding = self.t_embedding[:self.n_batches*self.batch_size]
        # create batches
        self.p_embedding_batch = np.split(self.p_embedding, self.n_batches, 0)
        self.p_length_batch = np.split(self.p_length, self.n_batches, 0)
        self.h_embedding_batch = np.split(self.h_embedding, self.n_batches, 0)
        self.h_length_batch = np.split(self.h_length, self.n_batches, 0)
        self.t_embedding_batch = np.split(self.t_embedding, self.n_batches, 0)

    def reset_batch(self):
        # build embeddings
        self.p_embedding = np.zeros((len(self.data['targets']), self.max_length), dtype=int)
        self.p_length = np.zeros((len(self.data['targets'])), dtype=int)
        self.h_embedding = np.zeros((len(self.data['targets']), self.max_length), dtype=int)
        self.h_length = np.zeros((len(self.data['targets'])), dtype=int)
        self.t_embedding = np.zeros((len(self.data['targets'])), dtype=int)
        self.prepro()
        self.build_batch()


    def next_batch(self):
        """Return current batch, increment pointer by 1 (modulo n_batches)"""
        p_embedding_batch = self.p_embedding_batch[self.pointer]
        p_length_batch = self.p_length_batch[self.pointer]
        h_embedding_batch = self.h_embedding_batch[self.pointer]
        h_length_batch = self.h_length_batch[self.pointer]
        t_embedding_batch = self.t_embedding_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.n_batches
        batch = {
            "premise":p_embedding_batch,
            "premise_length":p_length_batch,
            "hypothesis":h_embedding_batch,
            "hypothesis_length":h_length_batch,
            "target":t_embedding_batch
        }
        return batch
