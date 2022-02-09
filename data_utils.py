#!/usr/bin/env python
# coding: utf-8

import numpy as np

def parseDataFea(file, path='data/T98G/'):
    data_dict = {}
    idx_data_dict = {}
    idx_data_fea_dict = {}
    
    with open(path + file, 'r') as f:
        for i, line in enumerate(f.readlines()[1:]):
            
            line = line.strip(' \n ').split(',')
            name = line[0].strip('"')
            fea  = [float(i) for i in line[1:]]
            
            data_dict[name] = i
            idx_data_dict[i] = name
            idx_data_fea_dict[i] = fea
    
    return data_dict, idx_data_dict, idx_data_fea_dict


def parseA(file, a_dict, b_dict=None, path='data/T98G/'):
    if file == 'disease-disease_extract.csv':
        # <5.0
        # pos = [2, 3]
        
        # 5.0
        pos = [1, 2]
        dtype = np.uint8
    elif file == 'drugdrug_extract.csv':
        pos = [3, 4, 9]
        dtype = np.float32
    elif file == 'ppi_extract.csv':
        pos = [1, 2]
        dtype = np.uint8
    elif file == 'drugdisease_extract.csv':
        pos = [1, 2]
        dtype = np.uint8
    elif file == 'drugtarget_extract.csv':
        pos = [1, 2]
        dtype = np.uint8
    else:
        raise ValueError()

    if b_dict == None:
        b_dict = a_dict
        
    mat = np.zeros([len(a_dict), len(b_dict)], dtype=dtype)
    
    with open(path + file, 'r') as f:
        for line in f.readlines()[1:]:
            
            line = line.strip(' \n ').split(',')
            a = line[pos[0]].strip('"')
            b = line[pos[1]].strip('"')
            
            if a not in a_dict or b not in b_dict:
                continue
            
            if file != 'drugdrug_extract.csv':
                mat[a_dict[a], b_dict[b]] = 1
            else:
                mat[a_dict[a], b_dict[b]] = float(line[pos[2]])
    return mat

def parseA2(file, a_dict, path='data/T98G/'):
    if file != 'drugfeature1_finger_sim.csv' and file != 'drugfeature2_phychem_sim.csv':
        raise ValueError()
    
    mat = np.zeros([len(a_dict), len(a_dict)], dtype=np.float32)
    with open(path + file, 'r') as f:
        colums = []
        for i, line in enumerate(f.readlines()):
            line = line.strip('\n').split(',')
            if i == 0:
                colums = [a_dict[s.strip('"')] for s in line[1:]]
            else:
                cur = a_dict[line[0].strip('"')]
                for k, d in enumerate(line[1:]):
                    mat[cur, colums[k]] = d
    return mat