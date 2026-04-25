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
        logger.info("[Model Name]: " + str(current_file_name))
        self.logger = logger
        self.embedding_dim = embedding_dim
        self.rank = embedding_dim // 2
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device


        self.entity_emb = torch.nn.Embedding(num_entity, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relation, embedding_dim)
        self.type_emb = torch.nn.Embedding(2, embedding_dim * 3)

        self.context_vec = nn.Embedding(num_relation, self.rank)
        self.act = nn.Softmax(dim=1)


        self.init()
        self.x = 0.9
        self.y = 0.9

    def init(self):
        init_size = 0.01
        torch.nn.init.xavier_uniform_(self.entity_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.type_emb.weight.data)
        self.entity_emb.weight.data *= init_size
        self.relation_emb.weight.data *= init_size
        self.type_emb.weight.data *= init_size

        self.context_vec.weight.data = init_size * torch.randn_like(self.context_vec.weight.data)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_weights)

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())


    def cal_att_num_form(self, x, y, z, query):
        x, y, z, query = [i.view((-1, 1, self.rank)) for i in [x, y, z, query]]
        cands = torch.cat([x, y, z], dim=1)
        att_weight = torch.sum(query * cands / np.sqrt(self.rank), dim=-1, keepdim=True)
        att_weight = self.act(att_weight)
        return torch.sum(att_weight * cands, dim=1)


    def dvg(self, lhs, rel, rhs, ent_embs, view, rel_ids):
        lhs_re = lhs[:, :self.rank] + self.x * view[:, 2 * self.rank: 3 * self.rank]
        lhs_im = lhs[:, self.rank:] + self.x * view[:, 3 * self.rank: 4 * self.rank]

        rhs_re = rhs[:, :self.rank] + self.y * view[:, 4 * self.rank: 5 * self.rank]
        rhs_im = rhs[:, self.rank:] + self.y * view[:, 5 * self.rank: 6 * self.rank]

        view_re = view[:, :self.rank]
        view_im = view[:, self.rank: 2 * self.rank]
        rel_re = rel[:, :self.rank]
        rel_im = rel[:, self.rank:]
        right_re = ent_embs[:, :self.rank]
        right_im = ent_embs[:, self.rank:]


        rt_re_c = rel_re * view_re - rel_im * view_im
        rt_im_c = rel_re * view_im + rel_im * view_re


        rt_re_s = rel_re * view_re + rel_im * view_im
        rt_im_s = -(rel_re * view_im + rel_im * view_re)


        rt_re_d = rel_re * view_re
        rt_im_d = -(rel_re * view_im + rel_im * view_re)


        query = self.context_vec(rel_ids)
        fused_re = self.cal_att_num_form(rt_re_c, rt_re_s, rt_re_d, query)
        fused_im = self.cal_att_num_form(rt_im_c, rt_im_s, rt_im_d, query)


        final_re = lhs_re * fused_re - lhs_im * fused_im
        final_im = lhs_re * fused_im + lhs_im * fused_re


        pred = final_re @ right_re.t() + final_im @ right_im.t()


        norm_lhs = torch.sqrt(lhs_re ** 2 + lhs_im ** 2)
        norm_rel = torch.sqrt(fused_re ** 2 + fused_im ** 2)
        norm_rhs = torch.sqrt(rhs_re ** 2 + rhs_im ** 2)

        return pred, (norm_lhs, norm_rel, norm_rhs), self.type_emb.weight


    def forward(self, e1, rel, e2, batch_o):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e2 = self.to_var(e2)
        batch_o = self.to_var(batch_o)

        ent_embs = self.entity_emb.weight
        rel_embs = self.relation_emb.weight
        type_embs = self.type_emb.weight

        lhs = ent_embs[e1]
        rhs = ent_embs[e2]
        rel_emb = rel_embs[rel]
        typ = type_embs[batch_o]

        pred, a, b = self.dvg(lhs, rel_emb, rhs, ent_embs, typ, rel)
        return pred, a, b
