"""
Baseline LSTM model proposed in https://nlp.stanford.edu/pubs/snli_paper.pdf
Attention model introduced by https://arxiv.org/pdf/1509.06664.pdf
"""

import tensorflow as tf
import numpy as np


DATA_DIR = './data/'
GLOVE_FILE = 'glove.6B.300d.txt'
MODE = 'dev'

# data parameter
EMBEDDING_DIM = 300
MAX_LENGTH = 85
BATCH = 64

# model config
RNN_CELL = tf.contrib.rnn.GRUCell
DROP_CELL = tf.contrib.rnn.DropoutWrapper
NUM_UNIT = 60
LAYER = 1
ACTIVATION = tf.nn.tanh
NUM_LABELS = 3
LEARNING_RATE = 0.0005
DROP_OUT = 0.2
LAMBDA = 0.005

# attenntion
ATTENTION_CELL = tf.contrib.seq2seq.BahdanauAttention


class ConditionalEncoding(object):
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
            # concatenate the starting token
            start_tokens = tf.constant(0, shape=[BATCH], dtype=tf.int32) # starting token is at the 0th idex of word matrix
            self.hypothesis_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'hypothesis_input')
            hypothesis_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.hypothesis_inputs], 1)
            # update the sequence length to account for the additional token
            self.h_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')
            extra_length = tf.constant(1, shape=[BATCH], dtype=tf.int32)
            self.h_input_lengths = tf.add(self.h_input_lengths, extra_length)
            self.hypothesis_encode = tf.nn.embedding_lookup(self.word_matrix, hypothesis_inputs, name='h_encoded')

        # onehot encoding of labels
        with tf.variable_scope('output'):
            self.output = tf.placeholder(tf.int32, [BATCH], name = 'label')

        # embed premise with LSTM, retrieve the output of final state
        with tf.variable_scope('p_embed'):
            if LAYER > 1:
                p_cell = tf.contrib.rnn.MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT,output_keep_prob=1.0-DROP_OUT)) for _ in range(LAYER)])
            else:
                p_cell = RNN_CELL(NUM_UNIT)
                p_cell = DROP_CELL(p_cell, output_keep_prob=1.0 - DROP_OUT)
            init_state = p_cell.zero_state(BATCH, dtype=tf.float32)
            # BATCH x MAX_LENGTH x NUM_UNIT
            self.p_embedding, self.p_state = tf.nn.dynamic_rnn(p_cell, self.premise_encode, sequence_length=self.p_input_lengths, initial_state=init_state)

        # embed hypothesis with LSTM, retrieve the output of final state
        with tf.variable_scope('h_embed'):
            if LAYER > 1:
                h_cell = tf.contrib.rnn.MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT, output_keep_prob=1.0-DROP_OUT)) for _ in range(LAYER)])
            else:
                h_cell = RNN_CELL(NUM_UNIT)
                h_cell = DROP_CELL(h_cell, output_keep_prob=1.0-DROP_OUT)
            # BATCH x MAX_LENGTH x NUM_UNIT
            self.h_embedding, self.h_state = tf.nn.dynamic_rnn(p_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=self.p_state)


class BaselineLSTM(ConditionalEncoding):
    """
    CONDITIONAL ENCODING as suggested by Rocktaschel
    """
    def __init__(self):
        super(BaselineLSTM, self).__init__()

        # softmax layer, BATCH x FC_LAYER -> BATCH x NUM_LABELSs
        with tf.variable_scope('soft_max'):
            if RNN_CELL != tf.contrib.rnn.LSTMCell:
                units_in = self.h_state.get_shape().as_list()[-1]
                h_state = self.h_state
            else:
                units_in = self.h_state[0].get_shape().as_list()[-1]
                h_state = self.h_state[0]
            weights = tf.get_variable('w', shape=[units_in, NUM_LABELS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            biases = tf.get_variable('b', shape=[NUM_LABELS], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.logits = tf.nn.xw_plus_b(h_state, weights, biases)
            self.preds = tf.nn.softmax(self.logits)

        # calculate loss and optimize
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + LAMBDA*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        # mean accuracy of batch
        with tf.variable_scope('accuracy'):
            label = tf.cast(self.output, 'int32')
            corrects = tf.equal(tf.cast(tf.argmax(self.preds, 1), 'int32'), label)
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))


class WordbyWordAttention(ConditionalEncoding):
    """
    word by word attention as suggested by Rocktaschel
    """
    def __init__(self):
        super(WordbyWordAttention, self).__init__()

        if RNN_CELL != tf.contrib.rnn.LSTMCell:
            units_in = self.h_state.get_shape().as_list()[-1]
            h_state = self.h_state
        else:
            units_in = self.h_state[0].get_shape().as_list()[-1]
            h_state = self.h_state[0]

        with tf.variable_scope('attention'):
            attention = ATTENTION_CELL(NUM_UNIT, self.p_embedding, dtype=tf.float32)
            if LAYER > 1:
                h_cell = tf.contrib.rnn.MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT, output_keep_prob=1.0-DROP_OUT)) for _ in range(LAYER)])
            else:
                h_cell = RNN_CELL(NUM_UNIT)
                h_cell = DROP_CELL(h_cell, output_keep_prob=1.0-DROP_OUT)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(h_cell, attention)

            attention_state = attention_cell.zero_state(BATCH, dtype=tf.float32)
            attention_state = attention_state.clone(cell_state=self.p_state)

            # tf.nn.dynamic_rnn(attention_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=attention_state)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.h_embedding, sequence_length=self.h_input_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
            self.decoder_outputs, self.dec_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=MAX_LENGTH)

            # softmax layer, BATCH x FC_LAYER -> BATCH x NUM_LABELSs
            with tf.variable_scope('soft_max'):
                if RNN_CELL != tf.contrib.rnn.LSTMCell:
                    units_in = self.dec_state.cell_state.get_shape().as_list()[-1]
                    h_state = self.dec_state.cell_state
                else:
                    units_in = self.dec_state.cell_state[0].get_shape().as_list()[-1]
                    h_state = self.dec_state.cell_state[0]
                weights = tf.get_variable('w', shape=[units_in, NUM_LABELS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
                biases = tf.get_variable('b', shape=[NUM_LABELS], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
                self.logits = tf.nn.xw_plus_b(h_state, weights, biases)
                self.preds = tf.nn.softmax(self.logits)

            # calculate loss and optimize
            with tf.variable_scope('loss'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
                self.loss = tf.reduce_mean(cross_entropy) + LAMBDA*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            # mean accuracy of batch
            with tf.variable_scope('accuracy'):
                label = tf.cast(self.output, 'int32')
                corrects = tf.equal(tf.cast(tf.argmax(self.preds, 1), 'int32'), label)
                self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))


WordbyWordAttention()
