import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Doc2VecC(nn.Module):

    def __init__(self, vocab_size, emb_dim):
        super(Doc2VecC, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.center_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.context_embeddings = nn.Embedding(vocab_size, emb_dim)

        initrange = 1.0 / self.emb_dim
        init.uniform_(self.center_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.context_embeddings.weight.data, -initrange, initrange)

    def forward(self, center_w, local_context_w, global_context_w, negative_ws, lengths):

        emb_c = self.center_embeddings(center_w)
        emb_local = local_context_w.matmul(self.context_embeddings.weight)
        emb_global = global_context_w.matmul(self.context_embeddings.weight)
        emb_global /= lengths

        #Todo averaging is optional
        if True:
            emb_v = emb_local + emb_global

        neg_s = self.center_embeddings(negative_ws)
        pos_score = torch.sum(torch.mul(emb_c, emb_v), dim=1)
        pos_score = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_s, emb_v.unsqueeze(2)).squeeze()
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(pos_score + neg_score)

    def save_embedding(self, file_name):
        u_emb = self.context_embeddings.weight.cpu().data.numpy()
        #v_emb = self.context_embeddings.weight.cpu().data.numpy()

        word_embeddings = u_emb #+ v_emb
        word_embeddings.dump(file_name)
