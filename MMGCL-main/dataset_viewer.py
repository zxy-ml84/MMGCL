import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_load(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './dataset_sample/' + dataset
    train_edge = np.load(dir_str+'/train_sample.npy', allow_pickle=True)
    print(train_edge)
    print(train_edge.shape)
    user_item_dict = np.load(dir_str+'/user_item_dict_sample.npy', allow_pickle=True).item()
    # print(user_item_dict)
    if dataset == 'movielens':
        num_user = 100
        num_item = 1141
        v_feat = np.load(dir_str+'/v_feat_sample.npy', allow_pickle=True) if has_v else None
        print('v_feat', v_feat)
        print(v_feat.shape)
        a_feat = np.load(dir_str+'/a_feat_sample.npy', allow_pickle=True) if has_a else None
        print('a_feat', a_feat)
        print(a_feat.shape)
        t_feat = np.load(dir_str+'/t_feat_sample.npy', allow_pickle=True) if has_t else None
        print('t_feat', t_feat)
        print(t_feat.shape)
        v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
        a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None
    return num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat

class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user+num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])

# test = data_load('movielens')

