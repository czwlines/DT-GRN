import torch
import argparse
import numpy as np
from random import shuffle

import time
import random

from model import train
from data_utils import parseA, parseA2, parseDataFea
import os

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='GRN Model Sim Init')
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')
    parser.add_argument('--ckp_save_dir', type=str, default='./result/')
    parser.add_argument('--data_dir', type=str, default='./data/train/')
    parser.add_argument('--data_set', type=str, default='786-0')
    parser.add_argument('--model', type=str, default='[time]')
    parser.add_argument('--th_rate', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dru_hid_size', type=int, default=256)
    parser.add_argument('--dis_hid_size', type=int, default=256)
    parser.add_argument('--tar_hid_size', type=int, default=256)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--g_hid', type=int, default=1024)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--dp', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--print_epochs', type=int, default=10)
    parser.add_argument('--valid_epochs', type=int, default=100)
    parser.add_argument('--print_param_sum', type=bool, default=True)
    parser.add_argument('--dru_agg', type=str, default='edge')
    parser.add_argument('--decoder', type=str, default='Bilinear')
    parser.add_argument('--edge_mask', type=int, nargs='+', default=4)
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--cur_fold', type=int, default=0)
    return parser.parse_args()

# 计算 Threshold
def cal_threshold(mat, rate):
    for th in np.arange(0, 1, 0.01):
        cur_mat = mat > th
        if sum(cur_mat.sum(axis=0) > (cur_mat.shape[0] * rate)) == 0:
            return th
def norm(X):
    std1 = np.nanstd(X, axis=0)
    feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    X = np.tanh(X)
    means2 = np.mean(X, axis=0)
    std2 = np.std(X, axis=0)
    X = (X-means2)/std2
    X[:,std2==0]=0
    return torch.from_numpy(X)

def k_fold_data(mat, num_folds, cur_fold):
    data_items = torch.nonzero(mat).tolist()
    shuffle(data_items)
    fold_size = len(data_items) // num_folds
    valid_idx = np.arange(cur_fold * fold_size, (cur_fold + 1) * fold_size)

    train_data = torch.zeros_like(mat)
    valid_data = torch.zeros_like(mat)

    for e, (xi, xk) in enumerate(data_items):
        if e in valid_idx:
            valid_data[xi, xk] = mat[xi, xk]
        else:
            train_data[xi, xk] = mat[xi, xk]

    return train_data, valid_data

# 加载数据
def load_data(data_dir, data_set, rate, edge_mask, num_folds, cur_fold):
    # load data
    dru_dict, _, idx_dru_fea_dict1 = parseDataFea('drugfeature1_finger_extract.csv', path=data_dir + data_set + '/')
    _, _, idx_dru_fea_dict2 = parseDataFea('drugfeature2_phychem_sim.csv', path=data_dir + data_set + '/')
    dru_dru_mat = parseA('drugdrug_extract.csv', dru_dict, path=data_dir + data_set + '/')
    dru_dru_sim = parseA2('drugfeature2_phychem_sim.csv', dru_dict, path=data_dir + data_set + '/')
    
    dis_dict, _, idx_dis_fea_dict = parseDataFea('diseasefeature_extract3.csv', path=data_dir + data_set + '/')
    dis_dis_mat = parseA('disease-disease_extract.csv', dis_dict, path=data_dir + data_set + '/')
    
    tar_dict, _, idx_tar_fea_dict = parseDataFea('targetfeature_extract.csv', path=data_dir + data_set + '/')
    tar_tar_mat = parseA('ppi_extract.csv', tar_dict, path=data_dir + data_set + '/')
    
    dru_dis_mat = parseA('drugdisease_extract.csv', dru_dict, dis_dict, path=data_dir + data_set + '/')
    dru_tar_mat = parseA('drugtarget_extract.csv', dru_dict, tar_dict, path=data_dir + data_set + '/')

    print('-------------------  data info  -----------------')
    print('dru_size:', len(dru_dict))
    print('dru_fea_size:', len(idx_dru_fea_dict1[0]) + len(idx_dru_fea_dict2[0]))
    print('dis_size:', len(dis_dict))
    print('dis_fea_size:', len(idx_dis_fea_dict[0]))
    print('tar_size:', len(tar_dict))
    print('tar_fea_size:', len(idx_tar_fea_dict[0]))

    dru_dru_mat = torch.FloatTensor(dru_dru_mat)
    dru_dru_sim = torch.FloatTensor(dru_dru_sim)
    dis_dis_mat = torch.FloatTensor(dis_dis_mat)
    tar_tar_mat = torch.FloatTensor(tar_tar_mat)
    dru_dis_mat = torch.FloatTensor(dru_dis_mat)
    dru_tar_mat = torch.FloatTensor(dru_tar_mat)

    dru_dru_mat = score_norm(dru_dru_mat)
    train_dru_dru_mat, test_dru_dru_mat = k_fold_data(dru_dru_mat, num_folds, cur_fold)
    
    # mask
    train_mask = train_dru_dru_mat != 0
    valid_mask = test_dru_dru_mat != 0

    # calculate threshold
    threshold = cal_threshold(dru_dru_sim, rate)
    print('threshold:', threshold)
    
    # init adj
    dru_dru_A = torch.zeros_like(dru_dru_sim)
    dru_dru_A[dru_dru_sim > threshold] = 1
    # dru_dru_A[train_dru_dru_mat != 0] = 1
    # dru_dru_A[test_dru_dru_mat != 0] = 1
    dru_dru_A = dru_dru_A.long()

    dis_dis_A = torch.zeros_like(dis_dis_mat)
    dis_dis_A[dis_dis_mat != 0] = 2
    dis_dis_A = dis_dis_A.long()

    tar_tar_A = torch.zeros_like(tar_tar_mat)
    tar_tar_A[tar_tar_mat != 0] = 3
    tar_tar_A = tar_tar_A.long()

    dru_dis_A = torch.zeros_like(dru_dis_mat)
    dru_dis_A[dru_dis_mat != 0] = 4
    dru_dis_A = dru_dis_A.long()

    dru_tar_A = torch.zeros_like(dru_tar_mat)
    dru_tar_A[dru_tar_mat != 0] = 5
    dru_tar_A = dru_tar_A.long()
    
    if isinstance(edge_mask, int):
        edge_mask = [edge_mask]
    
    for mask in edge_mask:
        if mask == 1:
            dru_dru_A = torch.zeros_like(dru_dru_A)
            dru_dru_A = dru_dru_A.long()
            print(f'mask={mask}  reset dru_dru_A as zero, valid sum(dru_dru_A)={int(dru_dru_A.sum().item())}')
        elif mask == 2:
            dis_dis_A = torch.zeros_like(dis_dis_A)
            dis_dis_A = dis_dis_A.long()
            print(f'mask={mask}  reset dis_dis_A as zero, valid sum(dis_dis_A)={int(dis_dis_A.sum().item())}')
        elif mask == 3:
            tar_tar_A = torch.zeros_like(tar_tar_A)
            tar_tar_A = tar_tar_A.long()
            print(f'mask={mask}  reset tar_tar_A as zero, valid sum(tar_tar_A)={int(tar_tar_A.sum().item())}')
        elif mask == 4:
            dru_dis_A = torch.zeros_like(dru_dis_A)
            dru_dis_A = dru_dis_A.long()
            print(f'mask={mask}  reset dru_dis_A as zero, valid sum(dru_dis_A)={int(dru_dis_A.sum().item())}')
        elif mask == 5:
            dru_tar_A = torch.zeros_like(dru_tar_A)
            dru_tar_A = dru_tar_A.long()
            print(f'mask={mask}  reset dru_tar_A as zero, valid sum(dru_tar_A)={int(dru_tar_A.sum().item())}')

    # init feature
    dru_emb = torch.zeros(len(dru_dict), len(idx_dru_fea_dict1[0]) + len(idx_dru_fea_dict2[0]))
    for i in range(len(dru_dict)):
        dru_emb[i] = torch.FloatTensor(idx_dru_fea_dict1[i] + idx_dru_fea_dict2[i])
    dru_emb = norm(dru_emb)

    dis_emb = torch.zeros(len(dis_dict), len(idx_dis_fea_dict[0]))
    for i in range(len(dis_dict)):
        dis_emb[i] = torch.FloatTensor(idx_dis_fea_dict[i])
    dis_emb = norm(dis_emb)

    tar_emb = torch.zeros(len(tar_dict), len(idx_tar_fea_dict[0]))
    for i in range(len(tar_dict)):
        tar_emb[i] = torch.FloatTensor(idx_tar_fea_dict[i])
    tar_emb = norm(tar_emb)

    return \
        [dru_emb, dis_emb, tar_emb],\
        [dru_dru_A, dis_dis_A, tar_tar_A, dru_dis_A, dru_tar_A],\
        [train_dru_dru_mat, test_dru_dru_mat],\
        [train_mask, valid_mask]

def score_norm(dru_dru_mat):
    return dru_dru_mat / (dru_dru_mat.max() - dru_dru_mat.min())

def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()
    for key in args.__dict__:
        print(f"{key}: {args.__dict__[key]}")
    print('cur time: ', curtime())
    print('-------------------  mode: train  -----------------')
    set_seeds(args.seed)

    if args.model == '[time]':
        args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

    print('load data from: {} {}'.format(args.data_dir, args.data_set))
    print(f'train with {args.num_folds} fold')

    for cur_fold in range(args.num_folds):
        print(f'Start fold {cur_fold}/{args.num_folds}')
        data = load_data(args.data_dir, args.data_set, args.th_rate, args.edge_mask, args.num_folds, cur_fold)
        train(args, data, cur_fold)
        print(f'Finished fold {cur_fold}/{args.num_folds}')
