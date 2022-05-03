from __future__ import division
from __future__ import print_function
import sys
import os
import glob
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from LoadData import MyDataset
from models import SPAN_A
from torch.utils.data import DataLoader
from shutil import copyfile

def main(argv=None):
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--batchs', type=int, default=150, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')#5e-4
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--class_num', type=int, default=3, help='Class')
    parser.add_argument('--input_dim', type=int, default=33, help='Input dim')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 选择数据集
    Room_type = 'All_Room'


    # Load Data
    train_data=MyDataset(dataset=os.getcwd() + '/data/'+Room_type+'_train.json')
    val_data=MyDataset(dataset=os.getcwd() + '/data/'+Room_type+'_val.json')
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchs*8, shuffle=True ,num_workers=0,pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batchs, shuffle=True ,num_workers=0,pin_memory=True)

    # Model
    model = SPAN_A(args.input_dim,args.class_num,dropout=args.dropout,alpha=args.alpha,node_num = 14)

    if args.cuda:
        model.cuda()

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        print("{:-^50s}".format('epoch' + str(epoch+1)))

        loss_train_epoch = 0
        loss_val_epoch = 0
        acc_train_epoch = 0
        acc_val_epoch = 0

        t = time.time()
        # Train
        for i, (features, adjs, gmms, evaluates,scene_labels,zero_nodes,zero_masks) in enumerate(train_loader):
            evaluates = torch.reshape(evaluates,(-1,1)).squeeze()
            features, adjs, gmms, evaluates = Variable(features.cuda()), Variable(adjs.cuda()), Variable(gmms.cuda()), Variable(evaluates.cuda())
            model.train()
            optimizer.zero_grad()
            outputs = model(features, adjs)
            outputs = torch.reshape(outputs, (-1, args.class_num))
            loss_train = loss_fn(outputs, evaluates)
            loss_train.backward()
            optimizer.step()
            loss_train_epoch = loss_train_epoch + loss_train.data.item()

        loss_train_epoch = loss_train_epoch/len(train_loader)

        # Val
        if not args.fastmode:
            model.eval()

            for i, (features, adjs, gmms, evaluates,scene_labels,zero_nodes,zero_masks) in enumerate(val_loader):
                evaluates = torch.reshape(evaluates, (-1, 1)).squeeze()
                features, adjs, gmms, evaluates = Variable(features.cuda()), Variable(adjs.cuda()), Variable(gmms.cuda()), Variable(evaluates.cuda())
                outputs = model(features, adjs)
                outputs = torch.reshape(outputs, (-1, args.class_num))
                loss_val = loss_fn(outputs, evaluates)
                loss_val_epoch = loss_val_epoch + loss_val.data.item()

        loss_val_epoch = loss_val_epoch/len(val_loader)#计算出val_loss平均值
        acc_val_epoch = acc_val_epoch/len(val_loader)

        print('loss_train: {:.4f}'.format(loss_train_epoch),
              'loss_val: {:.4f}'.format(loss_val_epoch),
              'time: {:.4f}s'.format(time.time() - t))

        return loss_train_epoch,loss_val_epoch

    #训练模型
    t_total = time.time()
    train_loss = []#存储每个epoch训练集损失
    val_loss = []#存储每个epoch验证集损失
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    actual_epoch = 0

    for epoch in range(args.epochs):
        t_l,v_l = train(epoch)
        train_loss.append(t_l)
        val_loss.append(v_l)

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))#保存模型参数
        if val_loss[-1] < best:
            best = val_loss[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            actual_epoch = epoch
            break

        # 清除多余的模型参数
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        actual_epoch = epoch

    #清除多余的模型参数
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


    print('Loading {}th epoch'.format(best_epoch))
    copyfile(os.getcwd() + '/' + str(best_epoch)+'.pkl', os.getcwd() + '/model/3D-SPAN-A.pkl')
    os.remove(os.getcwd() + '/' + str(best_epoch)+'.pkl')


    x = np.array(range(actual_epoch+1))

    plt.title('Result')
    plt.plot(x,train_loss,color='blue',label='Train_Loss')
    plt.plot(x,val_loss,color='green',label='Val_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.legend()

    plt.show()


if __name__=='__main__':
    sys.exit(main())