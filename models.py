import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class SPAN_A(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, alpha, node_num):
        """Dense version of GAT."""
        super(SPAN_A, self).__init__()
        nheads = 4
        self.fc = nn.Linear(dim_in,dim_in)
        self.attentionlayer1 = [GraphAttentionLayer(dim_in, node_num, dropout=dropout, alpha=alpha, node_num=node_num, concat=True) for _ in range(nheads)]#中间层
        for i, attention in enumerate(self.attentionlayer1):
            self.add_module('attention_{}'.format(i), attention)
        self.attentionlayer2 = GraphAttentionLayer(node_num*nheads, node_num, dropout=dropout, alpha=alpha, node_num=node_num, concat=True)
        self.outputlayer = GraphAttentionLayer(node_num, dim_out, dropout=dropout, alpha=alpha, node_num=node_num, concat=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, adj):
        x = F.relu(self.fc(x))
        x = torch.cat([att(x, adj) for att in self.attentionlayer1], dim=2)
        x = self.dropout(x)
        x = self.attentionlayer2(x, adj)
        x = self.dropout(x)
        x = self.outputlayer(x, adj)
        y = self.dropout(x)

        # 多添加的Sofamax（记得删）
        y = torch.softmax(y,dim=2)

        return  y

class SPAN_B(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, alpha, node_num):
        """Dense version of GAT."""
        super(SPAN_B, self).__init__()
        self.node_num = node_num
        self.dim_out = dim_out
        self.fc = nn.Linear(dim_in,dim_in)
        self.attentionlayer = GraphAttentionLayer(dim_in, node_num, dropout=dropout, alpha=alpha, node_num=node_num,concat=True)
        self.outputlayer = GraphAttentionLayer(node_num, dim_out, dropout=dropout, alpha=alpha, node_num=node_num, concat=True)

        self.fc1 = nn.Linear(node_num*dim_out, node_num)
        self.fc2 = nn.Linear(node_num, dim_out)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.bn = nn.BatchNorm1d(12)
    def forward(self, x, l, adj):
        x = torch.cat((x, l), dim=2)
        x = F.relu(self.fc(x))

        x = self.attentionlayer(x, adj)
        x = self.dropout(x)
        x = self.outputlayer(x, adj)
        x = self.dropout(x)

        x = torch.reshape(x, (-1, self.node_num*self.dim_out))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        y = self.dropout(x)

        return  y
