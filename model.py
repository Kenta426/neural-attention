"""
Baseline LSTM model proposed in https://nlp.stanford.edu/pubs/snli_paper.pdf
Attention model introduced by https://arxiv.org/pdf/1509.06664.pdf
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.seq2seq import AttentionWrapper


# Global veriables
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
NUM_UNIT = 100
LAYER = 1
ACTIVATION = tf.nn.elu
NUM_LABELS = 3
LEARNING_RATE = 0.001
FC_LAYER = 30
DROP_OUT = 0.25
LAMBDA = 0.005

BI_DIRECTIONAL = False

# attenntion
ATTENTION_CELL = tf.contrib.seq2seq.LuongAttention


class ConditionalEncoding(object):
    """
    Module to pass 2 input texts and create conditional encoding
    """
    def __init__(self):
        """Creates TensorFlow graph of a conditional encoding.
        Args:
            TODO : Stop using global variables
        """

        # input premise -> vectorized premise
        with tf.variable_scope('premise'):
            # integer-encoded input premise
            self.premise_inputs = tf.placeholder(tf.int32, [BATCH, MAX_LENGTH], name = 'premise_input')
            # actual non-padded length of each input passages; used for dynamic unrolling
            self.p_input_lengths = tf.placeholder(tf.int32, [BATCH], name = 'input_length')

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

        with tf.variable_scope('output'):
            # correct labels for the inputs
            self.output = tf.placeholder(tf.int32, [BATCH], name = 'label')

        # embedding with pretrained glove
        with tf.variable_scope('embedding'):
            word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
            # make the embedding trainable during training
            # TODO: stop this feature at the test time
            self.word_matrix = tf.Variable(word_matrix, trainable=False, name='word_matrix')
            # embedding for the premise
            self.premise_encode = tf.nn.embedding_lookup(self.word_matrix, self.premise_inputs, name='p_encoded')
            # embedding for the hypothesis
            self.hypothesis_encode = tf.nn.embedding_lookup(self.word_matrix, hypothesis_inputs, name='h_encoded')

        # Create the embedding for hypothesis, with the context of premise (premise -> hypothesis)
        with tf.variable_scope('conditional_encoding'):
            # embed premise with LSTM, retrieve the output of final state
            with tf.variable_scope('p_embed'):
                if LAYER > 1:
                    p_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT),output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
                else:
                    p_cell = RNN_CELL(NUM_UNIT)
                    p_cell = DROP_CELL(p_cell, output_keep_prob=1.0-DROP_OUT)
                init_state = p_cell.zero_state(BATCH, dtype=tf.float32)
                # output has the shape of BATCH x MAX_LENGTH x NUM_UNIT
                self.p_embedding, self.p_state = tf.nn.dynamic_rnn(p_cell, self.premise_encode, sequence_length=self.p_input_lengths, initial_state=init_state)

                # tensorboard
                tf.summary.histogram('embedding', self.p_embedding)

            # embed hypothesis with LSTM, retrieve the output of final state
            with tf.variable_scope('h_embed'):
                if LAYER > 1:
                    h_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT), output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
                else:
                    h_cell = RNN_CELL(NUM_UNIT)
                    h_cell = DROP_CELL(h_cell, output_keep_prob=1.0-DROP_OUT)
                # output has the shape of BATCH x MAX_LENGTH x NUM_UNIT
                # set the initial state as the final state from premise
                self.h_embedding, self.h_state = tf.nn.dynamic_rnn(p_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=self.p_state)

                # tensorboard
                tf.summary.histogram('embedding', self.h_embedding)

        if BI_DIRECTIONAL:
            # Repeat the process with reversed order (hypothesis -> premise)
            with tf.variable_scope('conditional_encoding_reversed'):
                # embed premise with LSTM, retrieve the output of final state
                with tf.variable_scope('h_embed'):
                    if LAYER > 1:
                        h_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT),output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
                    else:
                        h_cell = RNN_CELL(NUM_UNIT)
                        h_cell = DROP_CELL(h_cell, output_keep_prob=1.0-DROP_OUT)
                    init_state = h_cell.zero_state(BATCH, dtype=tf.float32)
                    # output has the shape of BATCH x MAX_LENGTH x NUM_UNIT
                    self.h_embedding2, self.h_state2 = tf.nn.dynamic_rnn(p_cell, self.hypothesis_encode, sequence_length=self.h_input_lengths, initial_state=init_state)

                    # tensorboard
                    tf.summary.histogram('embedding', self.h_embedding2)

                # embed hypothesis with LSTM, retrieve the output of final state
                with tf.variable_scope('p_embed'):
                    if LAYER > 1:
                        p_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT), output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
                    else:
                        p_cell = RNN_CELL(NUM_UNIT)
                        p_cell = DROP_CELL(p_cell, output_keep_prob=1.0-DROP_OUT)
                    # output has the shape of BATCH x MAX_LENGTH x NUM_UNIT
                    # set the initial state as the final state from hypothesis
                    self.p_embedding2, self.p_state2 = tf.nn.dynamic_rnn(p_cell, self.premise_encode, sequence_length=self.p_input_lengths, initial_state=self.h_state2)

                    # tensorboard
                    tf.summary.histogram('embedding', self.p_embedding2)

class BaselineLSTM(ConditionalEncoding):
    """
    CONDITIONAL ENCODING as suggested by Rocktaschel
    """
    def __init__(self):
        # use conditional encoding
        super(BaselineLSTM, self).__init__()

        # connect the final state to fully connected layer
        with tf.variable_scope('fc_layer'):
            # retrieve the state from encoding
            if LAYER > 1:
                self.h_state = self.h_state[-1]
            if RNN_CELL != tf.contrib.rnn.LSTMCell:
                units_in = self.h_state.get_shape().as_list()[-1]
                h_state = self.h_state
            else:
                units_in = self.h_state[0].get_shape().as_list()[-1]
                h_state = self.h_state[0]
            # stack the final state if bi directional
            if BI_DIRECTIONAL:
                if LAYER > 1:
                    self.p_state2 = self.p_state2[-1]
                if RNN_CELL != tf.contrib.rnn.LSTMCell:
                    h_state2 = self.p_state2
                else:
                    h_state2 = self.p_state2[0]
                units_in = 2*units_in
                h_state = tf.concatenate(h_state, h_state2)
            # fully connected
            weights = tf.get_variable('w', shape=[units_in, FC_LAYER], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            biases = tf.get_variable('b', shape=[FC_LAYER], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            # add weights to regularlization term
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.fc_out = ACTIVATION(tf.nn.xw_plus_b(h_state, weights, biases))

            # tensorboard
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)

        # softmax layer, BATCH x FC_LAYER -> BATCH x NUM_LABELSs
        with tf.variable_scope('soft_max'):
            # map the output into probability
            weights = tf.get_variable('w', shape=[FC_LAYER, NUM_LABELS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            biases = tf.get_variable('b', shape=[NUM_LABELS], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            # add weights to regularlization term
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.logits = tf.nn.xw_plus_b(self.fc_out, weights, biases)

            # tensorboard
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)

        # calculate loss and optimize
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + LAMBDA*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # TODO: clip the gradient
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            # tensorboard
            tf.summary.scalar('loss', self.loss)

        # mean accuracy for batch
        with tf.variable_scope('accuracy'):
            self.preds = tf.nn.softmax(self.logits)
            label = tf.cast(self.output, 'int32')
            corrects = tf.equal(tf.cast(tf.argmax(self.preds, 1), 'int32'), label)
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

            # tensorboard
            tf.summary.scalar('accuracy', self.accuracy)


class WordbyWordAttention(ConditionalEncoding):
    """
    word by word attention as suggested by Rocktaschel
    """
    def __init__(self):
        # use conditional encoding
        super(WordbyWordAttention, self).__init__()

        # attend over the memory and read the input again
        with tf.variable_scope('attention'):
            # attention mechanisms over premise
            attention = ATTENTION_CELL(NUM_UNIT, self.p_embedding,
                memory_sequence_length=self.p_input_lengths, dtype=tf.float32)
            if LAYER > 1:
                a_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT), output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
            else:
                a_cell = RNN_CELL(NUM_UNIT)
                a_cell = DROP_CELL(a_cell, output_keep_prob=1.0-DROP_OUT)
            # attend over preise
            attention_cell = AttentionWrapper(a_cell, attention, alignment_history=True)
            attention_state = attention_cell.zero_state(BATCH, dtype=tf.float32)
            attention_state = attention_state.clone(cell_state=self.p_state)

            # dynamic decode over the hypothesis
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.h_embedding, sequence_length=self.h_input_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
            self.decoder_outputs, self.dec_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=MAX_LENGTH)
            # attention for visualizing heatmap
            self.dec_alignment = tf.transpose(self.dec_state.alignment_history.stack(), [1,0,2])

            # tensorboard
            tf.summary.histogram('attention', self.decoder_outputs.rnn_output)

        if BI_DIRECTIONAL:
            # repeaet the process if bi directional
            with tf.variable_scope('attention_reversed'):
                # attention mechanisms over hypothesis
                attention = ATTENTION_CELL(NUM_UNIT, self.h_embedding2,
                    memory_sequence_length=self.h_input_lengths, dtype=tf.float32)
                if LAYER > 1:
                    a_cell = MultiRNNCell([DROP_CELL(RNN_CELL(NUM_UNIT), output_keep_prob=1.0-DROP_OUT) for _ in range(LAYER)])
                else:
                    a_cell = RNN_CELL(NUM_UNIT)
                    a_cell = DROP_CELL(a_cell, output_keep_prob=1.0-DROP_OUT)
                # attend over hypothesis
                attention_cell = AttentionWrapper(a_cell, attention, alignment_history=True)
                attention_state = attention_cell.zero_state(BATCH, dtype=tf.float32)
                attention_state = attention_state.clone(cell_state=self.h_state)

                # dynamic decode over the premise
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.p_embedding2, sequence_length=self.p_input_lengths)
                decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
                self.decoder_outputs2, self.dec_state2, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=MAX_LENGTH)
                # attention for visualizing heatmap
                self.decoder_outputs2 = tf.transpose(self.dec_state2.alignment_history.stack(), [1,0,2])

                # tensorboard
                tf.summary.histogram('attention', self.decoder_outputs2.rnn_output)

        # if bidirectional, stack the 2 results
        with tf.variable_scope('combine'):
            if LAYER > 1:
                self.dec_state = self.dec_state[0][-1]
            else:
                self.dec_state = self.dec_state[0]
            if RNN_CELL != tf.contrib.rnn.LSTMCell:
                units_in = self.dec_state.get_shape().as_list()[-1]
                h_state = self.dec_state
            else:
                units_in = self.dec_state[0].get_shape().as_list()[-1]
                h_state = self.dec_state[0]
            # input to fully connected
            state = h_state

            if BI_DIRECTIONAL:
                if LAYER > 1:
                    self.dec_state2 = self.dec_state2[0][-1]
                else:
                    self.dec_state2 = self.dec_state2[0]
                if RNN_CELL != tf.contrib.rnn.LSTMCell:
                    h_state2 = self.dec_state2
                else:
                    h_state2 = self.dec_state2[0]
                # concatenate the result
                units_in = 2*units_in
                state = tf.concat([state,h_state2], 1)

        # connect the final state to fully connected layer
        with tf.variable_scope('fc_layer'):
            # fully connected
            weights = tf.get_variable('w', shape=[units_in, FC_LAYER], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            biases = tf.get_variable('b', shape=[FC_LAYER], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            # add weights to regularlization term
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.fc_out = ACTIVATION(tf.nn.xw_plus_b(state, weights, biases))

            # tensorboard
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)

        # softmax layer, BATCH x FC_LAYER -> BATCH x NUM_LABELSs
        with tf.variable_scope('soft_max'):
            weights = tf.get_variable('w', shape=[FC_LAYER, NUM_LABELS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.05))
            biases = tf.get_variable('b', shape=[NUM_LABELS], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

            # collect variables for the regularlization
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weights))
            self.logits = tf.nn.xw_plus_b(state, weights, biases)
            self.preds = tf.nn.softmax(self.logits)

            # tensorboard
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)

        # calculate loss and optimize
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + LAMBDA*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
            tf.summary.scalar('loss', self.loss)

        # mean accuracy of batch
        with tf.variable_scope('accuracy'):
            label = tf.cast(self.output, 'int32')
            corrects = tf.equal(tf.cast(tf.argmax(self.preds, 1), 'int32'), label)
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

BaselineLSTM()
