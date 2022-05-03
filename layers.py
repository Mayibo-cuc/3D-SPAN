import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, node_num,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.node_num = node_num
        self.fc= nn.Linear(in_features,out_features,bias=False)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))#注意力核
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.bn1 = nn.BatchNorm1d(self.node_num)
        self.bn2 = nn.BatchNorm1d(self.node_num)
        self.bn3 = nn.BatchNorm1d(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):#h为输入的N个点的特征向量，adj为邻接矩阵

        Wh = self.fc(h)
        a_input = self._prepare_attentional_mechanism_input(Wh)#数据预处理得到节点特征两两相连的矩阵，a_input: (batch,N,N, 2*out_features)
        attention = torch.matmul(a_input, self.a).squeeze()  # matmul是高维矩阵乘法,squeeze是去掉数量为1的维度，得到(batch,N,N)
        attention = self.bn1(attention.view(-1, self.node_num)).view(-1, self.node_num, self.node_num)
        attention = self.leakyrelu(attention)#根据邻接矩阵做Mask，得到注意力矩阵
        attention = self.bn2(attention.view(-1, self.node_num)).view(-1, self.node_num, self.node_num)
        attention = self.softmax(attention)#每行进行softmax归一化
        attention = self.dropout(attention)
        attention = attention * adj.cuda()

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            h_prime = self.bn3(h_prime.view(-1, self.out_features)).view(-1, self.node_num, self.out_features)
            return F.elu(h_prime)#激活函数，得到当前层的输出节点特征向量
        else:
            return h_prime#若是最后一层则直接输出

    def _prepare_attentional_mechanism_input(self, Wh):
        batch = Wh.size()[0]
        N = Wh.size()[1] # 节点数

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)#将tensor进行复制扩张
        Wh_repeated_alternating = Wh.repeat(1,N,1)#将tensor进行复制扩张（与上不同的复制扩张）


        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(batch,N, N, 2 * self.out_features)#改变tensor形状后返回

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
