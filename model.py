"""
Baseline LSTM model proposed in https://nlp.stanford.edu/pubs/snli_paper.pdf
Attention model introduced by https://arxiv.org/pdf/1509.06664.pdf
"""

import tensorflow as tf
import numpy as np


DATA_DIR = './data/'
GLOVE_FILE = 'glove.6b.300d.txt'
MODE = 'dev'

# data parameter
EMBEDDING_DIM = 300
MAX_LENGTH = 60
BATCH = 64

# model config
RNN_CELL = tf.contrib.rnn.LSTMCell
NUM_UNIT = 100
LAYER = 1
ACTIVATION = tf.nn.tanh
FC_LAYER = 200


class BaselineLSTM(object):
    def __init__(self):
        # pretrained glove
        word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
        self.word_matrix = tf.Variable(word_matrix, trainable=True, name='word_matrix')

        with tf.variable_scope('premise'):
            self.premise_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'premise_input')
            self.p_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')
            self.premise_encode = tf.nn.embedding_lookup(self.word_matrix, self.premise_inputs, name='p_encoded')

        with tf.variable_scope('hypothesis'):
            self.hypothesis_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'hypothesis_input')
            self.h_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')
            self.hypothesis_encode = tf.nn.embedding_lookup(self.word_matrix, self.hypothesis_inputs, name='h_encoded')

        with tf.variable_scope('output'):
            self.output = tf.placeholder(tf.float32, [BATCH, 3], name = 'one_hot')


        with tf.variable_scope('p_embed'):
            if LAYER > 1:
                p_cell = tf.contrib.rnn.MultiRNNCell([RNN_CELL(NUM_UNIT) for _ in range(LAYER)])
            else:
                p_cell = RNN_CELL(NUM_UNIT)
            init_state = p_cell.zero_state(BATCH, dtype=tf.float32)
            # BATCH x MAX_LENGTH x NUM_UNIT
            self.premise_embed, _ = tf.nn.dynamic_rnn(p_cell, self.premise_encode, sequence_length=self.p_input_lengths, initial_state=init_state)

        with tf.variable_scope('h_embed'):
            if LAYER > 1:
                h_cell = tf.contrib.rnn.MultiRNNCell([RNN_CELL(NUM_UNIT) for _ in range(LAYER)])
            else:
                h_cell = RNN_CELL(NUM_UNIT)
            init_state = h_cell.zero_state(BATCH, dtype=tf.float32)
            # BATCH x MAX_LENGTH x NUM_UNIT
            self.hypothesis_embed, _ = tf.nn.dynamic_rnn(p_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=init_state)

        with tf.variable_scope('fc'):
            inputs = tf.concat([self.hypothesis_embed, self.premise_embed], 1)
            print(inputs.get_shape())
        # with tf.variable_scope('')


BaselineLSTM()
