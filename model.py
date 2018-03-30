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
FC_LAYER = 50
NUM_LABELS = 4
LEARNING_RATE = 0.001
LAMBDA = 0.0001


class BaselineLSTM(object):
    """
    CONDITIONAL ENCODING as suggested by Rocktaschel
    """
    def __init__(self):
        # pretrained glove
        word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
        self.word_matrix = tf.Variable(word_matrix, trainable=True, name='word_matrix')

        # input premise -> vectorized premise
        with tf.variable_scope('premise'):
            self.premise_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'premise_input')
            self.p_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')
            self.premise_encode = tf.nn.embedding_lookup(self.word_matrix, self.premise_inputs, name='p_encoded')

        # input hypothesis -> vectorized hypothesis
        with tf.variable_scope('hypothesis'):
            self.hypothesis_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'hypothesis_input')
            self.h_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')
            self.hypothesis_encode = tf.nn.embedding_lookup(self.word_matrix, self.hypothesis_inputs, name='h_encoded')

        # onehot encoding of labels
        with tf.variable_scope('output'):
            self.output = tf.placeholder(tf.int32, [BATCH, NUM_LABELS], name = 'one_hot')

        # embed premise with LSTM, retrieve the output of final state
        with tf.variable_scope('p_embed'):
            if LAYER > 1:
                p_cell = tf.contrib.rnn.MultiRNNCell([RNN_CELL(NUM_UNIT) for _ in range(LAYER)])
            else:
                p_cell = RNN_CELL(NUM_UNIT)
            init_state = p_cell.zero_state(BATCH, dtype=tf.float32)
            # BATCH x MAX_LENGTH x NUM_UNIT
            _, premise_state = tf.nn.dynamic_rnn(p_cell, self.premise_encode, sequence_length=self.p_input_lengths, initial_state=init_state)

        # embed hypothesis with LSTM, retrieve the output of final state
        with tf.variable_scope('h_embed'):
            if LAYER > 1:
                h_cell = tf.contrib.rnn.MultiRNNCell([RNN_CELL(NUM_UNIT) for _ in range(LAYER)])
            else:
                h_cell = RNN_CELL(NUM_UNIT)
            # BATCH x MAX_LENGTH x NUM_UNIT
            self.hypothesis_embed,_ = tf.nn.dynamic_rnn(p_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=premise_state)
            # get the last output
            self.hypothesis_embed = tf.transpose(self.hypothesis_embed, [1,0,2])[-1]

        # fully connected layer 1, BATCH x 2*NUM_UNIT -> BATCH x FC_LAYER
        with tf.variable_scope('fc1'):
            inputs = self.hypothesis_embed
            units_in = inputs.get_shape().as_list()[-1]
            weights = tf.get_variable('w', shape=[units_in, FC_LAYER], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05), )
            biases = tf.get_variable('b', shape=[FC_LAYER], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            self.fc1 = ACTIVATION(tf.nn.xw_plus_b(inputs, weights, biases))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))

        # softmax layer, BATCH x FC_LAYER -> BATCH x NUM_LABELSs
        with tf.variable_scope('soft_max'):
            units_in = self.fc1.get_shape().as_list()[-1]
            weights = tf.get_variable('w', shape=[units_in, NUM_LABELS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05), )
            biases = tf.get_variable('b', shape=[NUM_LABELS], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.logits = tf.nn.xw_plus_b(self.fc1, weights, biases)
            self.preds = tf.nn.softmax(self.logits)

        # mean accuracy of batch
        with tf.variable_scope('accuracy'):
            label = tf.cast(tf.argmax(self.output, 1), 'int32')
            corrects = tf.equal(tf.cast(tf.argmax(self.preds, 1), 'int32'), label)
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # calculate loss and optimize
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=[self.logits])
            self.loss = tf.reduce_mean(cross_entropy) + LAMBDA*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

BaselineLSTM()
