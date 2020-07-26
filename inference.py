import json

import numpy as np
from tqdm import tqdm

from embedding_projector import normalize_rows


def inference(texts, word_vecs, word2id):
    id2doc = {}
    domains = sorted(list(texts.keys()))

    dim = word_vecs.shape[1]
    doc_vecs = np.zeros(shape=(len(domains), dim))

    for i, d in tqdm(enumerate(domains)):

        id2doc[i] = d
        w_ids = [word2id[w] for w in texts[d].strip().lower().split(' ') if w in word2id]
        vec = np.sum(word_vecs[w_ids], axis=0)
        if any(w_ids):
            doc_vecs[i] = vec / len(w_ids)

    doc_vecs = normalize_rows(doc_vecs)

    doc_vecs.dump("embeddings/doc_vectors.npy")
    json.dump(id2doc, open("embeddings/index2doc.json", 'w'))


if __name__ == "__main__":
    id2word = json.load(open('embeddings/index2word.json', 'r'))
    word2id = {w: int(i) for i, w in id2word.items()}

    word_vecs = np.load("embeddings/word_vectors.npy", allow_pickle=True)
    texts = json.load(open('data/all_texts.json'))
    inference(texts, word_vecs, word2id)
