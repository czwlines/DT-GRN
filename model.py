#!/usr/bin/env python
# coding: utf-8
# author: clines

import numpy as np

import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# GRUCell
class GRUCell(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)

        self.c = nn.Linear(x_dim, h_dim, True)
        self.u = nn.Linear(h_dim, h_dim, True)

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)
        # print('x + h: ', rz_input.size())
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(x) + r * self.u(h))

        new_h = z * h + (1 - z) * u
        return new_h

# GRU
class GRU(nn.Module):
    def __init__(self, dru_emb, dis_emb, tar_emb, dru_hid, dis_hid, tar_hid, edge_dim, g_hid):
        '''
            dru_emb: 811
            dis_emb: 935
            tar_emb: 953

            dru_hid: 811
            dis_hid: 935
            tar_hid: 953

            edge_dim: 50

            g_hid: 1000
        '''
        super(GRU, self).__init__()

        # dru_gru: 881 + 881 + (322 + 50) + (1437 + 50) + 1000, 811
        self.dru_gru = GRUCell(dru_emb + dru_hid + edge_dim + dis_hid + edge_dim + tar_hid + g_hid, dru_hid)

        # dis_gru: 935 + 935 + (881 + 50) + 1000, 935
        self.dis_gru = GRUCell(dis_emb + dis_hid + edge_dim + dru_hid + g_hid, dis_hid)

        # tar_gru: 1437 + 1437 + (881 + 50) + 1000, 953
        self.tar_gru = GRUCell(tar_emb + tar_hid + edge_dim + dru_hid + g_hid, tar_hid)

        #   g_gru: 881 + 322 + 1437, 1000
        self.g_gru = GRUCell(dru_hid + dis_hid + tar_hid, g_hid)

    def forward(self, i, h, g, mask):
        '''
            dru_i: [811, 881 + 811 + (322 + 50) + (1437 + 50)]
            dis_i: [935, 322 + 322 + (881 + 50)]
            tar_i: [953, 1437 + 1437 + (881 + 50)]

            dru_h: [811, 881]
            dis_h: [935, 322]
            tar_h: [953, 1437]

            dru_m: [811, 811]
            dis_m: [935, 935]
            tar_m: [953, 953]

                g: [1000]
        '''
        dru_i, dis_i, tar_i = i
        dru_h, dis_h, tar_h = h
        dru_m, dis_m, tar_m = mask

        # g_expand_dru: [811, 1000]
        g_expand_dru = g.unsqueeze(0).expand(dru_h.size(0), g.size(-1))
        # x: [811, 881 + 811 + (322 + 50) + (1437 + 50) + 1000]
        x = torch.cat((dru_i, g_expand_dru), -1)
        # new_dru_h: [811, 881]
        new_dru_h = self.dru_gru(x, dru_h)

        # g_expand_dru: [935, 1000]
        g_expand_dis = g.unsqueeze(0).expand(dis_h.size(0), g.size(-1))
        # x: [935, 322 + 322 + (881 + 50) + 1000]
        x = torch.cat((dis_i, g_expand_dis), -1)
        # new_dis_h: [935, 322]
        new_dis_h = self.dis_gru(x, dis_h)

        # g_expand_dru: [953, 1000]
        g_expand_tar = g.unsqueeze(0).expand(tar_h.size(0), g.size(-1))
        # x: [953, 1437 + 1437 + (881 + 50) + 1000]
        x = torch.cat((tar_i, g_expand_tar), -1)
        # new_tar_h: [953, 1437]
        new_tar_h = self.tar_gru(x, tar_h)

        # [881]
        dru_h_mean = new_dru_h.sum(0) / new_dru_h.size(0)
        # [322]
        dis_h_mean = new_dis_h.sum(0) / new_dis_h.size(0)
        # [1437]
        tar_h_mean = new_tar_h.sum(0) / new_tar_h.size(0)

        # [881 + 322 + 1437]
        mean = torch.cat((dru_h_mean, dis_h_mean, tar_h_mean), -1)
        new_g = self.g_gru(mean, g)

        return new_dru_h, new_dis_h, new_tar_h, new_g

# GRN
class GRNGOB(nn.Module):
    def __init__(self, dru_emb, dis_emb, tar_emb, dru_hid, dis_hid, tar_hid, edge_dim, g_hid, dp=0.1, layer=2,
                 agg='gate', dru_agg='edge', device='cpu'):
        super(GRNGOB, self).__init__()
        self.layer = layer
        self.dp = dp

        self.slstm = GRU(dru_emb, dis_emb, tar_emb, dru_hid, dis_hid, tar_hid, edge_dim, g_hid)

        self.dru_hid = dru_hid
        self.dis_hid = dis_hid
        self.tar_hid = tar_hid
        self.g_hid = g_hid

        self.agg = agg
        self.dru_agg = dru_agg
        self.device = device

        # dru-dru, dru-dis, dru-tar, dis-dis, tar-tar
        self.edgeemb = nn.Embedding(6, edge_dim)

        # dru-dis
        self.gate1 = nn.Linear(dru_hid + dis_hid + edge_dim, dru_hid + edge_dim)
        self.gate2 = nn.Linear(dru_hid + dis_hid + edge_dim, dis_hid + edge_dim)

        # dru-tar
        self.gate3 = nn.Linear(dru_hid + tar_hid + edge_dim, dru_hid + edge_dim)
        self.gate4 = nn.Linear(dru_hid + tar_hid + edge_dim, tar_hid + edge_dim)
        
        # dru_agg param
        if self.dru_agg != 'edge':
            self.weight = nn.Linear(dru_emb, dru_emb)

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    def forward(self, dru_emb, dis_emb, tar_emb, mat):
        '''
            dru_emb: [811, 881]
            dis_emb: [935, 322]
            tar_emb: [953, 1437]


            mat: dru_dru_mat, dis_dis_mat, tar_tar_mat, dru_dis_mat, dru_tar_mat
                value:
                    0: none
                    1: dru-dru
                    2: dis-dis
                    3: tar-tar
                    4: dru-dis
                    5: dru-tar
        '''

        dru_size = dru_emb.size(0)
        dis_size = dis_emb.size(0)
        tar_size = tar_emb.size(0)

        dru_dru_mat, dis_dis_mat, tar_tar_mat, dru_dis_mat, dru_tar_mat = mat

        # [811, 811]
        dru_dru_mask = (dru_dru_mat != 0).float()
        dru_dru_mask_t = dru_dru_mask.transpose(0, 1)

        # [935, 935]
        dis_dis_mask = (dis_dis_mat != 0).float()
        dis_dis_mask_t = dis_dis_mask.transpose(0, 1)

        # [953, 953]
        tar_tar_mask = (tar_tar_mat != 0).float()
        tar_tar_mask_t = tar_tar_mask.transpose(0, 1)

        # [811, 935]
        dru_dis_mask = (dru_dis_mat != 0).float()
        dru_dis_mask_t = dru_dis_mask.transpose(0, 1)

        # [811, 953]
        dru_tar_mask = (dru_tar_mat != 0).float()
        dru_tar_mask_t = dru_tar_mask.transpose(0, 1)

        # [*, *, 50]
        dru_dis_edge = self.edgeemb(dru_dis_mat)
        dru_tar_edge = self.edgeemb(dru_tar_mat)

        dru_dis_edge_t = dru_dis_edge.transpose(0, 1)
        dru_tar_edge_t = dru_tar_edge.transpose(0, 1)

        if self.dru_agg == 'edge':
            dru2dru = (dru_dru_mat == 1).float()
        else:
            dru2dru = dru_dru_mat.float()
        
        dis2dis = (dis_dis_mat == 2).float()
        tar2tar = (tar_tar_mat == 3).float()

        dru_h = torch.zeros_like(dru_emb)
        dis_h = torch.zeros_like(dis_emb)
        tar_h = torch.zeros_like(tar_emb)
        g_h = dru_emb.new_zeros(self.g_hid)

        for i in range(self.layer):

            if self.dru_agg == 'edge':
                dru_neigh_dru_h = self.sum(dru_h, dru2dru)
            else:
                dru_neigh_dru_h = self.weight(self.sum(dru_h, dru2dru))

            dis_neigh_dis_h = self.sum(dis_h, dis2dis)

            tar_neigh_tar_h = self.sum(tar_h, tar2tar)

            if self.agg == 'gate':
                # dis -> dru
                # [811, 935, 322]
                dis_expand_h = dis_h.unsqueeze(0).expand(dru_size, dis_size, self.dis_hid)
                # [811, 935, 881]
                dru_expand_h = dru_h.unsqueeze(1).expand(dru_size, dis_size, self.dru_hid)
                # [811, 935, 322 + 50]
                dis_h_expand_edge = torch.cat((dis_expand_h, dru_dis_edge), -1)
                # [811, 935, 881 + 322 + 50]
                dis_dru_expand_edge = torch.cat((dis_h_expand_edge, dru_expand_h), -1)
                # [811, 935, 322 + 50]
                g2 = torch.sigmoid(self.gate2(dis_dru_expand_edge))
                # [811, 935, 322 + 50]
                dru_neigh_dis_h = dis_h_expand_edge * g2 * dru_dis_mask.unsqueeze(2)
                # [811, 322 + 50]
                dru_neigh_dis_h = dru_neigh_dis_h.sum(1)

                # dru -> dis
                # [935, 811, 881]
                dru_expand_h = dru_h.unsqueeze(0).expand(dis_size, dru_size, self.dru_hid)
                # [935, 811, 322]
                dis_expand_h = dis_h.unsqueeze(1).expand(dis_size, dru_size, self.dis_hid)
                # [935, 811, 881 + 50]
                dru_h_expand_edge = torch.cat((dru_expand_h, dru_dis_edge_t), -1)
                # [935, 811, 881 + 322 + 50]
                dru_dis_expand_edge = torch.cat((dru_h_expand_edge, dis_expand_h), -1)
                # [935, 811, 881 + 50]
                g1 = torch.sigmoid(self.gate1(dru_dis_expand_edge))
                # [935, 811, 881 + 50]
                dis_neigh_dru_h = dru_h_expand_edge * g1 * dru_dis_mask_t.unsqueeze(2)
                # [935, 881 + 50]
                dis_neigh_dru_h = dis_neigh_dru_h.sum(1)

                # dru -> tar
                # [953, 811, 881]
                dru_expand_h = dru_h.unsqueeze(0).expand(tar_size, dru_size, self.dru_hid)
                # [953, 811, 881]
                tar_expand_h = tar_h.unsqueeze(1).expand(tar_size, dru_size, self.tar_hid)
                # [953, 811, 881 + 50]
                dru_h_expand_edge = torch.cat((dru_expand_h, dru_tar_edge_t), -1)
                # [953, 811, 881 + 1437 + 50]
                dru_dru_expand_edge = torch.cat((dru_h_expand_edge, tar_expand_h), -1)
                # [953, 811, 881 + 50]
                g3 = torch.sigmoid(self.gate3(dru_dru_expand_edge))
                # [953, 811, 881 + 50]
                tar_neigh_dru_h = dru_h_expand_edge * g3 * dru_tar_mask_t.unsqueeze(2)
                # [953, 881 + 50]
                tar_neigh_dru_h = tar_neigh_dru_h.sum(1)

                # tar -> dru
                # [811, 953, 1437]
                tar_expand_h = tar_h.unsqueeze(0).expand(dru_size, tar_size, self.tar_hid)
                # [811, 953, 881]
                dru_expand_h = dru_h.unsqueeze(1).expand(dru_size, tar_size, self.dru_hid)
                # [811, 953, 1437 + 50]
                tar_h_expand_edge = torch.cat((tar_expand_h, dru_tar_edge), -1)
                # [811, 953, 881 + 1437 + 50]
                tar_dru_expand_edge = torch.cat((tar_h_expand_edge, dru_expand_h), -1)
                # [811, 953, 1437 + 50]
                g4 = torch.sigmoid(self.gate4(tar_dru_expand_edge))
                # [811, 953, 1437 + 50]
                dru_neigh_tar_h = tar_h_expand_edge * g4 * dru_tar_mask.unsqueeze(2)
                # [811, 1437 + 50]
                dru_neigh_tar_h = dru_neigh_tar_h.sum(1)

            # [811, 881 + 811 + (322 + 50) + (1437 + 50)]
            dru_input = torch.cat((dru_emb, dru_neigh_dru_h, dru_neigh_dis_h, dru_neigh_tar_h), -1)
            # [935, 322 + 322 + (881 + 50)]
            dis_input = torch.cat((dis_emb, dis_neigh_dis_h, dis_neigh_dru_h), -1)
            # [953, 1437 + 1437 + (881 + 50)]
            tar_input = torch.cat((tar_emb, tar_neigh_tar_h, tar_neigh_dru_h), -1)

            dru_h, dis_h, tar_h, g_h = self.slstm((dru_input, dis_input, tar_input), (dru_h, dis_h, tar_h), g_h,
                                                  (dru_dru_mask, dis_dis_mask, tar_tar_mask))

        if self.dp > 0:
            dru_h = F.dropout(dru_h, self.dp, self.training)

        return dru_h, g_h

class BilinearDecoder(nn.Module):
    def __init__(self, input_size):
        super(BilinearDecoder, self).__init__()
        self.weight = nn.Linear(input_size, input_size, bias=False)
    
    def forward(self, zu, zv):
        zu = zu.view(1, -1)
        zv = zv.view(1, -1)
        intermediate_product = self.weight(zu)
        ret = torch.matmul(intermediate_product, zv.reshape(-1, 1))
        return torch.tanh(ret)

class FFNDecoder(nn.Module):
    def __init__(self, input_size):
        super(FFNDecoder, self).__init__()
        self.weight = nn.Linear(input_size * 2, 1, bias=False)
    
    def forward(self, zu, zv):
        zu = zu.view(1, -1)
        zv = zv.view(1, -1)
        ret = self.weight(torch.cat([zu, zv], -1))
        return torch.tanh(ret)
    
# Model
class Model(nn.Module):
    def __init__(self,
                 dru_emb,
                 dis_emb,
                 tar_emb,
                 dru_hid,
                 dis_hid,
                 tar_hid,
                 edge_dim,
                 g_hid,
                 decoder='FFN',
                 dp=0.1,
                 layer=2,
                 agg='gate',
                 dru_agg='edge',
                 device='cpu'
                 ):
        super(Model, self).__init__()

        self.dp = dp

        self.dru_linear = nn.Linear(dru_emb, dru_hid)
        self.dis_linear = nn.Linear(dis_emb, dis_hid)
        self.tar_linear = nn.Linear(tar_emb, tar_hid)

        self.encoder = GRNGOB(dru_hid, dis_hid, tar_hid, dru_hid, dis_hid, tar_hid, edge_dim, g_hid, dp, layer, agg, dru_agg, device)
        
        if decoder == 'FFN':
            self.decoder = FFNDecoder(dru_hid)
        else:
            self.decoder = BilinearDecoder(dru_hid)
        
#         print(self.decoder)

    def forward(self, dru_emb, dis_emb, tar_emb, dru_dru_mat, dis_dis_mat, tar_tar_mat, dru_dis_mat, dru_tar_mat, mask):
        dru_emb = F.relu(self.dru_linear(dru_emb))
        dis_emb = F.relu(self.dis_linear(dis_emb))
        tar_emb = F.relu(self.tar_linear(tar_emb))

        if self.dp > 0:
            dru_emb = F.dropout(dru_emb, self.dp, self.training)
            dis_emb = F.dropout(dis_emb, self.dp, self.training)
            tar_emb = F.dropout(tar_emb, self.dp, self.training)

        dru_h, g_h = self.encoder(dru_emb, dis_emb, tar_emb,
                                  (dru_dru_mat, dis_dis_mat, tar_tar_mat, dru_dis_mat, dru_tar_mat))

        r = torch.zeros_like(dru_dru_mat).float()

        for i, k in torch.nonzero(mask):
            r[i, k] = self.decoder(dru_h[i], dru_h[k])

        return r

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, ipt, target, mask):
        ipt = ipt.contiguous().view(1, -1)
        target = target.contiguous().view(1, -1)
        mask = mask.contiguous().view(1, -1).float()

        ipt = ipt * mask
        target = target * mask

        return self.loss(ipt, target)

def calculate_pearsonr(ipt, tar, mask):
    a, b = [], []
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    return pearsonr(a, b)[0]

def calculate_spearmanr(ipt, tar, mask):
    a, b = [], []
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    return spearmanr(a, b)[0]

def calculate_r2score(ipt, tar, mask):
    a, b = [], []
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    return r2_score(b, a)
    
def print_total_param(model):
    total_params = 0
    for name, parameters in model.named_parameters():
        params = np.prod(list(parameters.size()))
        total_params += params
    print('total parameters: {:.4f}M'.format(total_params / 1e6))

def train(args, data, cur_fold=None):

    emb, A, target, mask = data
    dru_emb, dis_emb, tar_emb = emb
    dru_dru_A, dis_dis_A, tar_tar_A, dru_dis_A, dru_tar_A = A
    train_target, valid_target = target
    train_mask, valid_mask = mask

    device = args.device

    dru_emb = dru_emb.to(device)
    dis_emb = dis_emb.to(device)
    tar_emb = tar_emb.to(device)

    dru_dru_A = dru_dru_A.to(device)
    dis_dis_A = dis_dis_A.to(device)
    tar_tar_A = tar_tar_A.to(device)
    dru_dis_A = dru_dis_A.to(device)
    dru_tar_A = dru_tar_A.to(device)

    train_target = train_target.to(device)
    valid_target = valid_target.to(device)

    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)

    model = Model(
        dru_emb=len(dru_emb[0]),
        dis_emb=len(dis_emb[0]),
        tar_emb=len(tar_emb[0]),
        dru_hid=args.dru_hid_size,
        dis_hid=args.dis_hid_size,
        tar_hid=args.tar_hid_size,
        edge_dim=args.edge_dim,
        g_hid=args.g_hid,
        layer=args.layer,
        dp=args.dp,
        decoder=args.decoder,
        dru_agg=args.dru_agg,
        device=args.device
    )
    model = model.to(device)
    print_total_param(model)

    loss_fn = MaskLoss()

    optimizer = torch.optim.Adam(model.parameters())

    best_epoch = 1
    best_loss = np.inf

    t1 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        pred = model(dru_emb, dis_emb, tar_emb, dru_dru_A, dis_dis_A, tar_tar_A, dru_dis_A, dru_tar_A, train_mask)

        loss = loss_fn(pred, train_target, train_mask)
        pear = calculate_pearsonr(pred, train_target, train_mask)
        spea = calculate_spearmanr(pred, train_target, train_mask)
        r2sc = calculate_r2score(pred, train_target, train_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.print_epochs == 0:
            t2 = time.time()
            print(f'train epc:{epoch}    loss:{loss.item():.4f}    pear:{pear:.4f}    spea:{spea:.4f}    r2:{r2sc:.4f}    t:{(t2 - t1) / args.print_epochs:.8f}')
            t1 = time.time()

        if epoch % args.valid_epochs == 0:
            print('-------- valid --------')
            with torch.no_grad():
                model.eval()

                pred = model(dru_emb, dis_emb, tar_emb, dru_dru_A, dis_dis_A, tar_tar_A, dru_dis_A, dru_tar_A, valid_mask)

                loss = loss_fn(pred, valid_target, valid_mask)
                pear = calculate_pearsonr(pred, valid_target, valid_mask)
                spea = calculate_spearmanr(pred, valid_target, valid_mask)
                r2sc = calculate_r2score(pred, valid_target, valid_mask)

                print(f'valid epc:{epoch}    loss:{loss.item():.4f}    pear:{pear:.4f}    spea:{spea:.4f}    r2:{r2sc:.4f}')

                for i, k in torch.nonzero(valid_target)[:20]:
                    print('%s %s pred: %s    true: %s' % (i.item(), k.item(), pred[i, k].item(), valid_target[i, k].item()))

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch

                if cur_fold == None or not hasattr(args, 'num_folds'):
                    save_path = '{}/{}{}_{}.best.pt'.format(args.ckp_save_dir, args.model, args.data_set.strip('/'), args.edge_mask)
                else:
                    save_path = '{}/{}{}_{}_{}-{}.best.pt'.format(args.ckp_save_dir, args.model, args.data_set.strip('/'), args.edge_mask, args.num_folds, cur_fold)
                    
                print('save best model at {}'.format(save_path))
                checkpoint = {
                    'model': model.state_dict(),
                    'args': args,
                    'loss': best_loss
                }
                torch.save(checkpoint, save_path)

    return best_loss, best_epoch