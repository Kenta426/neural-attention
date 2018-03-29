""" Attention model introduced by 'https://arxiv.org/pdf/1509.06664.pdf'
"""


import tensorflow as tf

DATA_DIR = './data/'
GLOVE_FILE = 'glove.6b.300d.txt'
MODE = 'dev'
EMBEDDING_DIM = 300 

class Attention(object):
    def __init__(self):

        with tf.variable_scope('premise'):
            pass

        with tf.variable_scope('hypothesis'):
            pass
