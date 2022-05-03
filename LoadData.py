import numpy as np
import scipy.sparse as sp
import torch
import os
import csv
import json
import math
import pickle
import random
from torch.utils.data import Dataset

def accuracy_F1(output, labels, zero_num):# 用于三分类的准确率和召回率计算
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()

    cor = np.zeros(output.shape[1])
    num = np.zeros(output.shape[1])
    for i in range(len(labels)):
        num[labels[i]] = num[labels[i]]+1
        if correct[i] == 1:
            cor[labels[i]] = cor[labels[i]] + 1

    return cor,num


def Extract_data(filepath):
    features = []
    adjs = []
    gmms = []
    evaluates = []
    scene_label = []
    zero_node = []
    zero_mask= []
    scene_names = []

    with open(filepath, 'r') as f:
        data = json.load(f)  # 直接从json文件中读取数据返回一个python对象
    f.close()
    graphs = data['graph']

    for i in range(len(graphs)):
        scene_names.append(graphs[i]['scene_name'])
        features.append(graphs[i]['feature'])
        adjs.append(graphs[i]['adj'])
        gmms.append(graphs[i]['gmm'])
        zero_node.append([graphs[i]['zero_num']])
        zero_mask.append([graphs[i]['zero_mask']])
        scene_score = graphs[i]['scene_score']

        scene_label.append([int(scene_score)])

        evaluate_content = graphs[i]['evaluate']

        evaluates.append(evaluate_content)

    return features,adjs,gmms,evaluates,scene_label,zero_node,zero_mask,scene_names


class MyDataset(Dataset):
    def __init__(self, dataset,transform=None,target_transform=None):
        super(MyDataset, self).__init__()

        features, adjs, gmms, evaluates,scene_labels,zero_node,zero_mask,scene_names = Extract_data(dataset)

        self.features = features
        self.adjs = adjs
        self.gmms = gmms
        self.evaluates = evaluates
        self.scene_labels = scene_labels
        self.zero_nodes = zero_node
        self.zero_masks = zero_mask

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature = self.features[index]
        adj = self.adjs[index]
        gmm = self.gmms[index]
        evaluate = self.evaluates[index]
        scene_label = self.scene_labels[index]
        zero_nodes = self.zero_nodes[index]
        zero_masks = self.zero_masks[index]

        feature = torch.FloatTensor(np.array(feature))
        adj = torch.FloatTensor(np.array(adj))
        gmm = torch.FloatTensor(np.array(gmm))
        evaluate = torch.LongTensor(np.array(evaluate))
        scene_label = torch.LongTensor(np.array(scene_label))
        zero_nodes = np.array(zero_nodes)
        zero_masks = torch.FloatTensor(np.array(zero_masks))
        return feature, adj, gmm, evaluate,scene_label,zero_nodes,zero_masks

    def __len__(self):
            return len(self.features)

