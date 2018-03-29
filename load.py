"""
Load required data
"""

import spacy
import pandas as pd
import os
import model
import numpy as np
from tqdm import tqdm

NLP = spacy.blank('en')

def tokenize(text):
    """tokenize input text
    Arg:
        text (str): string to tokenize
    Returns:
        list: tokenized sentence
    """
    return [i.text for i in NLP(text)]

def load_glove(path):
    """Load pre-trained GloVe vectors into dictionary.
    Args:
        path (str): path to .txt file containing pre-trained GloVe vectors.
    Returns:
        dict:       dictionary mapping word to its vector representation.
    """
    print ("\nLoading glove:")
    # took 30s on my laptop
    embedding_vectors = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            line_split = line.strip().split(' ')
            vector = np.array(line_split[1:], dtype=float)
            word = line_split[0]
            embedding_vectors[word] = vector
    return embedding_vectors

def process_sentences(df):
    """Tokenize all sentences in the data and return vocabulary
    Args:
        df (dict): dict that maps keys ("premises", "hypothesis", "target")
                   to the np.array of sentences.
                   Only tokenize "premise" and "hypothesis"
    Returns:
        dict:      same format as the input dict, but tokenized
        dict:      all vocab in the training set
    """
    vocab = set([])
    for sent_type in ["premises", "hypothesis"]:
        tokenized_data = []
        for sent in tqdm(df[sent_type]):
            token = tokenize(sent[0])
            tokenized_data.append(token)
            vocab.update(set(token))
        df[sent_type] = tokenized_data
    return df, vocab

def build_wordmatrix(glove, vocab):
    """Build a word vector matrix from pretrained model and vocab in the documents
    initialize out-of-vocabulary as a random vector
    Args:
        glove (dict): dict that maps word to pretrained vector
        vocab (set):  set of vocabulary found in the training documents
    Returns:
        np.ndarray:   embedding matrix with shape (vocab size+1, embedding size)
        dict:         dict that maps vocabulary word to id (row in the embedding matrix)
        dict:         dict that maps id to word
    """
    word2id = {}
    id2word = {}
    # vocab + 1 for the EOF token
    embedding_matrix = np.zeros([len(vocab)+1, model.EMBEDDING_DIM])

    glove_vocab = glove.keys()
    for i,v in enumerate(vocab):
        if v in glove_vocab:
            word2id[v] = i
            id2word[i] = v
            embedding_matrix[i] = glove[v]
        else:
            word2id[v] = i
            id2word[i] = v
            # TODO: training time, the out-of-vocab is randomly initialized but also optimized, too
            embedding_matrix[i] = np.random.uniform(-0.05, 0.05, model.EMBEDDING_DIM)
    word2id["EOF"] = i+1
    id2word[i+1] = "EOF"
    return embedding_matrix, word2id, id2word

def load_data():
    """Load dataset and convert it to tokenized sentences, generate an embedding matrix,
    Output mapping dictionary
    """
    #TODO: this process should go inside of Loader class or Batcher
    
    vocab = {}
    print("\nLoading dataset:")
    df = pd.read_csv(os.path.join(model.DATA_DIR, "snli_1.0_{}.txt".format(model.MODE)), delimiter="\t")
    dataset = {
        "premises": df[["sentence1"]].values,
        "hypothesis": df[["sentence2"]].values,
        "targets": df[["gold_label"]].values}
    sentences, vocab = process_sentences(dataset)
    glove = load_glove(os.path.join(model.DATA_DIR, "glove.6B", model.GLOVE_FILE))
    embedding_matrix, word2id, id2word = build_wordmatrix(glove, vocab)
    return sentences, embedding_matrix, (word2id, id2word)

load_data()
