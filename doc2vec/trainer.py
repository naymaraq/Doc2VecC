import json
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doc2vec.dataset import WebTexts, Doc2vecDataset
from doc2vec.model import Doc2VecC
from doc2vec.utils import load_cfg


class Doc2VecTrainer:

    def __init__(self, config_path="configs/config.yaml"):

        args = load_cfg(config_path)
        self.args = args
        self.data = WebTexts(args)
        self.dataset = Doc2vecDataset(self.data)

        self.id2token = self.dataset.data_.id2token
        self.dataloader = DataLoader(self.dataset, batch_size=args["batch_size"],
                                     shuffle=True, num_workers=6, collate_fn=self.dataset.collate)

        self.vocab_size = len(self.data.token2id)
        self.emb_dim = args["emb_dim"]

        self.batch_size = args["batch_size"]
        self.iterations = args["epochs"]
        self.initial_lr = args["initial_lr"]

        self.doc2vec_model = Doc2VecC(self.vocab_size, self.emb_dim)
        self.use_cuda = torch.cuda.is_available() and args["use_cuda"]
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.doc2vec_model.cuda()

    def train(self):

        optimizer = optim.AdamW(self.doc2vec_model.parameters(), lr=self.initial_lr)
        path_to_save = os.path.join(self.args["output_folder"], "word_vectors.npy")

        json.dump(self.id2token, open(os.path.join(self.args["output_folder"], "index2word.json"), 'w'))
        for iteration in range(self.iterations):

            running_loss = 0.0
            for i, sample_batched in tqdm(enumerate(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    c = sample_batched[0].to(self.device)
                    l_c = sample_batched[1].to(self.device)
                    g_c = sample_batched[2].to(self.device)
                    neg_s = sample_batched[3].to(self.device)
                    t = sample_batched[4].to(self.device)

                    optimizer.zero_grad()

                    loss = self.doc2vec_model.forward(c, l_c, g_c, neg_s, t)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1

            if (iteration + 1) % self.args["print_every"] == 0:
                print("Iter {}: Loss: {}".format(iteration, running_loss))

            self.doc2vec_model.save_embedding(path_to_save)


tr = Doc2VecTrainer()
tr.train()
