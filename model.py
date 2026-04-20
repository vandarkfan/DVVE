import torch
import os
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class DVVE(torch.nn.Module):
    def __init__(self, logger, num_entity, num_relation,
                 embedding_dim=300, device='cuda:0', ):
        super().__init__()
        current_file_name = os.path.basename(__file__)
        logger.info( "[Model Name]: " + str(current_file_name))
        self.logger = logger
        self.embedding_dim = embedding_dim
        self.rank = embedding_dim // 2
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device
        self.entity_emb = torch.nn.Embedding(num_entity, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relation, embedding_dim)
        self.type_emb = torch.nn.Embedding(2, embedding_dim*3)
        self.init()


    def init(self):
        init_size = 0.01
        torch.nn.init.xavier_uniform_(self.entity_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.type_emb.weight.data)
        self.entity_emb.weight.data *= init_size
        self.relation_emb.weight.data *= init_size
        self.type_emb.weight.data *= init_size
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_weights)

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())

    def DVC(self, lhs, rel, rhs,ent_embs, view):

        lhs = lhs[:, :self.rank] + view[:, 2 * self.rank:3 * self.rank], lhs[:, self.rank:] + view[:,
        3 * self.rank:4 * self.rank]
        rhs = rhs[:, :self.rank] + view[:, 4 * self.rank:5 * self.rank], rhs[:, self.rank:] + view[:,
        5 * self.rank:6 * self.rank]
        bias_t_r =  view[:, 4 * self.rank:5 * self.rank]
        bias_t_i =  view[:, 5 * self.rank:6 * self.rank]

        view = view[:, :self.rank], view[:, self.rank:2 * self.rank]

        right = ent_embs
        right = right[:, :self.rank], right[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2 * self.rank]

        rt = rel[0] * view[0], rel[1] * view[0], rel[0] * view[1], rel[1] * view[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r, 1, keepdim=True) +
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t() + torch.sum(
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i, 1, keepdim=True)
        ),(
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        ), self.type_emb.weight


    def forward(self, e1, rel, e2, batch_o):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e2 = self.to_var(e2)
        ent_embs = self.entity_emb.weight
        rel_embs = self.relation_emb.weight
        type_embs = self.type_emb.weight
        lhs = ent_embs[e1]  # [batch_size, embedding_dim]
        rhs = ent_embs[e2]  # [num_entity, embedding_dim]
        rel = rel_embs[rel]
        typ = type_embs[batch_o]
        pred,a,b = self.DVC(lhs, rel, rhs,ent_embs,typ)

        return pred,a,b

