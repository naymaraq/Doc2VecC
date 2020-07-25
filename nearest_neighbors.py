import argparse
import json
import os

import numpy as np


def normalize_rows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape((N, 1)) + 1e-30
    return x


class NN:
    def nn_search(self, word):
        pass

    def load_vectors(self):
        self.db = np.load(self.vectors_path, allow_pickle=True)
        self.db = normalize_rows(self.db)

    def load_vocab(self):
        self.id2token = json.load(open(self.vocab_path, 'r'))
        self.id2token = {int(i): w for i, w in self.id2token.items()}
        self.token2id = {w: i for i, w in self.id2token.items()}

    def get_query_vec(self, token):
        ix = self.token2id[token]
        return self.db[ix].reshape(1, -1)


class NaiveNN(NN):

    def __init__(self, topk, embedding_folder='./embeddings', doc_or_word="word"):
        self.topk = topk
        self.vectors_path = os.path.join(embedding_folder, "{}_vectors.npy".format(doc_or_word))
        self.vocab_path = os.path.join(embedding_folder, "index2{}.json".format(doc_or_word))

        self.load_vocab()
        self.load_vectors()

    def nn_search(self, q_token):
        cosine = lambda a, db: np.dot(a, self.db.T)
        vec = self.get_query_vec(q_token)
        scores = cosine(vec, self.db)[0]
        indecies = scores.argsort()[-self.topk:][::-1]
        D = [self.id2token[i] for i in indecies]
        I = [scores[i] for i in indecies]
        I = np.squeeze(I)
        D = np.squeeze(D)
        return D, I


class AproximateNN(NN):

    def __init__(self, topk, embedding_folder='./embeddings', doc_or_word="word"):
        self.topk = topk
        self.vectors_path = os.path.join(embedding_folder, "{}_vectors.npy".format(doc_or_word))
        self.vocab_path = os.path.join(embedding_folder, "index2{}.json".format(doc_or_word))

        self.load_vectors()
        self.load_vocab()

    def load_vectors(self):
        self.db = np.load(self.vectors_path, allow_pickle=True)
        self.db = normalize_rows(self.db)
        self.dim = self.db.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.db = np.ascontiguousarray(self.db, dtype=np.float32)
        self.index.add(self.db)

    def nn_search(self, q_token):
        vec = self.get_query_vec(q_token)
        I, D = self.index.search(vec.reshape(1, -1), self.topk)
        I = np.squeeze(I)
        D = np.squeeze(D)
        D = [self.id2token[i] for i in D]
        return D, I


nn = NaiveNN
try:
    import faiss

    nn = AproximateNN
except:
    pass

parser = argparse.ArgumentParser(description='NN Query')

parser.add_argument('-topk', type=int, default=5)
parser.add_argument('-query', type=str)
parser.add_argument('-doc', type=str)

args = parser.parse_args()

if __name__ == "__main__":

    nn_obj = nn(args.topk + 1, doc_or_word='doc' if args.doc else 'word')
    most_similars, scores = nn_obj.nn_search(args.query)

    i = 0
    for k, v in zip(most_similars, scores):
        if i == 0:
            i += 1
            continue
        print('Top {} nearest: {}, score {}'.format(i, k, round(v, 4)))
        i += 1
