from __future__ import division
from __future__ import print_function
import sys
import os
import random
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from LoadData import MyDataset,accuracy_F1
from models import SPAN_A
from torch.utils.data import DataLoader

def main(argv=None):
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--batchs', type=int, default=300, help='Batch size.It needs to be as large as the test set.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')#5e-4
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=200, help='Patience')
    parser.add_argument('--class_num', type=int, default=3, help='Class')
    parser.add_argument('--input_dim', type=int, default=33, help='Input dim')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load Data
    test_data = MyDataset(dataset=os.getcwd() + '/data/All_Room_test.json')

    test_loader = DataLoader(dataset=test_data, batch_size=args.batchs, shuffle=False,num_workers=0,pin_memory=True)

    # Model
    model = SPAN_A(args.input_dim,args.class_num,dropout=args.dropout,alpha=args.alpha,node_num = 14)
    model.load_state_dict(torch.load('./model/3D-SPAN-A.pkl'))
    if args.cuda:
        model.cuda()

    # Test
    def compute_test():
        model.eval()
        cor_all = np.zeros(args.class_num).tolist()
        num_all = np.zeros(args.class_num).tolist()

        for i, (features, adjs, gmms, evaluates,scene_labels,zero_nodes,zero_masks) in enumerate(test_loader):
            evaluates = torch.reshape(evaluates, (-1, 1)).squeeze()
            features, adjs, gmms, evaluates = Variable(features.cuda()), Variable(adjs.cuda()), Variable(gmms.cuda()), Variable(evaluates.cuda())
            outputs = model(features, adjs)
            outputs = torch.reshape(outputs, (-1, args.class_num))
            cor,num = accuracy_F1(outputs, evaluates, zero_nodes.sum().item())


            cor_all = cor_all + cor
            num_all = num_all + num

        all_acc = cor_all.sum()/num_all.sum()
        rc = cor_all/num_all
        print(rc)
        recall = rc.sum() / args.class_num
        F1 = 2 * recall * all_acc / (recall + all_acc)
        print("Test set recall= {:.4f}".format(recall))
        print("Test set accuracy= {:.4f}".format(all_acc))
        print("Test set F1= {:.4f}".format(F1))

    compute_test()

if __name__=='__main__':
    sys.exit(main())