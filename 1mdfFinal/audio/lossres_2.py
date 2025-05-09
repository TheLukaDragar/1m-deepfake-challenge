import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np


class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.th_similar_min = 0.9
        self.th_different_max = 0.1

    def cosine_similarity(self, x1, x2, eps=1e-8):
        '''
        pair-wise cosine distance
        x1: [M, D]
        x2: [N, D]
        similarity: [M, N]
        '''
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        similarity = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return similarity

    def forward(self, embeddings, length, label):
        loss_batch = 0
        num_batch = embeddings.size()[0]
        num_batch_dynamic = num_batch
        for ibat in range(num_batch):
            embedding = embeddings[ibat, :, :]

            # ori_length = int(length[ibat, :])
            # true_label = label[ibat, 0:ori_length]  # obtain the true length before zero padding
            true_label = label[ibat, :]
            assert true_label.size()[0] == 64
            real_mask = torch.where(true_label == 1)  # real frame position
            fake_mask = torch.where(true_label == 0)  # fake frame position
            emb_dim = 32
            Real_embedding = torch.empty([emb_dim, 0]).cuda()
            Fake_embedding = torch.empty([emb_dim, 0]).cuda()
            # scalenum = int(1050 / 132)  # ratio of label sequence to the embedding sequence

            wind = 64
            vec = 64
            scalenum = vec // wind #aka 4
            
            for i in real_mask[0]:
                proportion = int(i.item()) / wind
                start = int(vec * proportion)
                s_emb = embedding[:, i].cuda()
                s_emb = torch.unsqueeze(s_emb, dim=1)
                emb = torch.empty([emb_dim, 0]).cuda()
                for j in range(start, start + scalenum):
                    s_emb = embedding[:, j].cuda()
                    s_emb = torch.unsqueeze(s_emb, dim=1)
                    emb = torch.cat([emb, s_emb], dim=1).cuda()
                Real_embedding = torch.cat([Real_embedding, emb], dim=1)  #concat all real embedding frames
            for i in fake_mask[0]:
                proportion = int(i.item()) / wind
                start = int(vec * proportion)
                emb = torch.empty([emb_dim, 0]).cuda()
                for j in range(start, start + scalenum):
                    s_emb = embedding[:, j].cuda()
                    s_emb = torch.unsqueeze(s_emb, dim=1)
                    emb = torch.cat([emb, s_emb], dim=1).cuda()
                Fake_embedding = torch.cat([Fake_embedding, emb], dim=1)  #concat all fake embedding frames

            # print(Real_embedding.size(), 'real_size')
            # print(Fake_embedding.size(), 'fake_size')

            r_embedding = Real_embedding
            r_embedding = r_embedding.t()  # [M, D]
            f_embedding = Fake_embedding
            f_embedding = f_embedding.t()
            if Real_embedding.size()[1] == 0:  # if no real frames, all fake embeddings should be similar
                sim_f2f = self.cosine_similarity(f_embedding, f_embedding)
                sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
                zero = torch.zeros_like(sim_f2f_hard)
                loss_f2f = torch.max(self.th_similar_min - sim_f2f_hard, zero)
                loss_f2f = loss_f2f.mean()
                continue
            if Fake_embedding.size()[1] == 0:  # if no fake frames, all real embeddings should be similar
                sim_r2r = self.cosine_similarity(r_embedding, r_embedding)
                sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
                zero = torch.zeros_like(sim_r2r_hard)
                loss_r2r = torch.max(self.th_similar_min - sim_r2r_hard, zero)
                loss_r2r = loss_r2r.mean()
                continue
              # all fake embedings should be similar
            sim_f2f = self.cosine_similarity(f_embedding, f_embedding)
            sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
            zero = torch.zeros_like(sim_f2f_hard)
            loss_f2f = torch.max(self.th_similar_min - sim_f2f_hard, zero)
            loss_f2f = loss_f2f.mean()

            # all real embeddings should be similar
            sim_r2r = self.cosine_similarity(r_embedding, r_embedding)
            sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
            zero = torch.zeros_like(sim_r2r_hard)
            loss_r2r = torch.max(self.th_similar_min - sim_r2r_hard, zero)
            loss_r2r = loss_r2r.mean()

            # fake embeddings should be different with real embeddings
            # sim_f2r = self.cosine_similarity(f_embedding, r_embedding)
            # # f2r
            # sim_f2r_hard = torch.max(sim_f2r, dim=1)[0]
            # zero = torch.zeros_like(sim_f2r_hard)
            # loss_f2r = torch.max(sim_f2r_hard - self.th_different_max, zero)
            # loss_f2r = loss_f2r.mean()
            # # r2f
            # sim_b2f_hard = torch.max(sim_f2r, dim=0)[0]
            # zero = torch.zeros_like(sim_r2f_hard)
            # loss_r2f = torch.max(sim_r2f_hard - self.th_different_max, zero)
            # loss_r2f = loss_r2f.mean()

            sim_f2r = self.cosine_similarity(f_embedding, r_embedding)
            # f2r
            sim_f2r_hard = torch.max(sim_f2r, dim=1)[0]
            zero = torch.zeros_like(sim_f2r_hard)
            loss_f2r = torch.max(sim_f2r_hard - self.th_different_max, zero)
            loss_f2r = loss_f2r.mean()

            # r2f
            sim_b2f_hard = torch.max(sim_f2r, dim=0)[0]
            zero = torch.zeros_like(sim_b2f_hard)
            loss_r2f = torch.max(sim_b2f_hard - self.th_different_max, zero)
            loss_r2f = loss_r2f.mean()

            loss_batch = loss_batch + loss_f2f + loss_r2r + loss_f2r + loss_r2f

        loss_batch = loss_batch / num_batch_dynamic
        return loss_batch



