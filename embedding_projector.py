import json
import os

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


def project(word_vecs, id2word):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        f.write("{}\t{}\n".format("index", "word"))
        for i, word in id2word.items():
            f.write("{}\t{}\n".format(i, word))

    weights = tf.Variable(word_vecs)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def normalize_rows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x

import argparse

parser = argparse.ArgumentParser(description='Embedding Projector')

parser.add_argument('-vecs', type=str)
parser.add_argument('-vocab', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    id2word = json.load(open(args.vocab, 'r'))
    word_vecs = np.load(args.vecs, allow_pickle=True)
    word_vecs = normalize_rows(word_vecs)
    project(word_vecs, id2word)
