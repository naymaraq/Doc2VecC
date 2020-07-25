import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class WebTexts:
    """
    Stanford Sentiment Treebank reader
    """

    def __init__(self, config):

        self.input_file = config["input_file"]
        self.window_size = config["window_size"]
        self.k = config["num_of_negatives"]
        self.q = 1 - config["curruption_rate"]

        self.sentences = []
        self.sentences_count = 0
        self.all_sentences = []
        self.sampling_probs = []
        self.sample_table = []

        self.token2id = dict()
        self.id2token = dict()
        self.tokenfreq = dict()
        self.token_count = 0

        self.tablesize = 10000000
        self.negpos = 0

    def get_negatives(self):

        sample_table = self.get_sample_table()
        response = sample_table[self.negpos:self.negpos + self.k]
        self.negpos = (self.negpos + self.k) % len(sample_table)
        if len(response) != self.k:
            return np.concatenate((response, sample_table[0:self.negpos]))
        return response

    def get_contexts(self, idx):

        C = self.window_size
        allsent = self.get_all_sentences()

        sent_id = idx % len(allsent)
        sent = allsent[sent_id]
        word_id = random.randint(0, len(sent) - 1)

        l_context = sent[max(0, word_id - C): word_id]
        if word_id + 1 < len(sent):
            l_context += sent[word_id + 1: min(len(sent), word_id + C + 1)]

        centerword = sent[word_id]
        l_context = [w for w in l_context if w != centerword]
        g_context = np.random.choice(sent, size=int(self.q * len(sent)), replace=False)
        g_context = [w for w in g_context if w != centerword]
        if any(l_context) and any(g_context):
            return centerword, l_context, g_context, len(sent)
        else:
            return self.get_contexts(idx + 1)

    def get_sample_table(self):

        if hasattr(self, 'sample_table') and self.sample_table:
            return self.sample_table

        token2id = self.tokens()
        n_tokens = len(token2id)
        sampling_freq = np.zeros((n_tokens,))

        for w, i in token2id.items():
            freq = self.tokenfreq.get(w, 0) ** 0.75
            sampling_freq[i] = freq

        sampling_freq /= np.sum(sampling_freq)
        sampling_freq = np.cumsum(sampling_freq) * self.tablesize

        self.sample_table = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > sampling_freq[j]:
                j += 1
            self.sample_table[i] = j

        np.random.shuffle(self.sample_table)
        return self.sample_table

    def num_sentences(self):
        if hasattr(self, "sentences_count") and self.sentences_count:
            return self.sentences_count
        else:
            self.sentences_count = len(self.get_all_sentences())
            return self.sentences_count

    def sub_sampling_prob(self):

        if hasattr(self, 'sampling_probs') and any(self.sampling_probs):
            return self.sampling_probs

        threshold = self.token_count * 0.7

        n_tokens = len(self.token2id)
        sampling_probs = np.zeros((n_tokens,))
        for w, i in self.token2id.items():
            freq = self.tokenfreq[w]
            sampling_probs[i] = np.sqrt(freq / threshold) + (freq / threshold)

        self.sampling_probs = sampling_probs
        return self.sampling_probs

    def get_all_sentences(self):

        if hasattr(self, 'all_sentences') and any(self.all_sentences):
            return self.all_sentences

        sentences = self.get_sentences()
        token2id = self.tokens()
        sampling_probs = self.sub_sampling_prob()

        allsentences = [[w for w in s if w in token2id and random.random() >= sampling_probs[token2id[w]]] for s in
                        sentences]
        allsentences = [s for s in allsentences if len(s) > 1]

        self.all_sentences = allsentences

        return self.all_sentences

    def get_sentences(self):

        if hasattr(self, 'sentences') and any(self.sentences):
            return self.sentences

        sentences = []

        texts = json.load(open(self.input_file))
        for d, text in texts.items():
            splitted = text.strip().split(' ')
            sentences += [[w.lower() for w in splitted]]

        self.sentences = sentences
        return self.sentences

    def tokens(self):

        if hasattr(self, 'token2id') and len(self.token2id):
            return self.token2id

        for sentence in self.get_sentences():
            if len(sentence) > 1:
                for w in sentence:
                    if len(w) > 0:
                        self.token_count += 1
                        self.tokenfreq[w] = self.tokenfreq.get(w, 0) + 1

        idx = 0
        for w, c in sorted(self.tokenfreq.items(), key=lambda x: x[1], reverse=True):

            if c < 20:
                continue
            self.token2id[w] = idx
            self.id2token[idx] = w
            # self.tokenfreq[idx] = c
            idx += 1

        return self.token2id


class Doc2vecDataset(Dataset):

    def __init__(self, text_data):
        self.data_ = text_data
        self.token2id = self.data_.tokens()
        self.vocab_size = len(self.token2id)

    def __len__(self):
        return self.data_.num_sentences()

    def __getitem__(self, idx):
        samples = []
        centerword, l_context, g_context, t = self.data_.get_contexts(idx)

        #center word index
        c = self.token2id[centerword]

        #local context BoW
        l = [self.token2id[w] for w in l_context]
        l_c = np.zeros(self.vocab_size)
        l_c[l] = 1.

        #global context BoW
        g = [self.token2id[w] for w in g_context]
        g_c = np.zeros(self.vocab_size)
        g_c[g] = 1./ self.data_.q

        #negative samples indexes
        n_s = self.data_.get_negatives()
        samples.append((c, l_c, g_c, n_s, [t]))
        return samples

    @staticmethod
    def collate(batches):
        all_c, all_l_c, all_g_c, all_n_s, all_t = [], [], [], [], []
        for batch in batches:
            for c, l_c, g_c, neg_v, t in batch:
                all_c.append(c)
                all_l_c.append(l_c)
                all_g_c.append(g_c)
                all_n_s.append(neg_v)
                all_t.append(t)

        return torch.LongTensor(all_c), torch.FloatTensor(all_l_c), \
               torch.FloatTensor(all_g_c), torch.LongTensor(all_n_s), torch.LongTensor(all_t)
