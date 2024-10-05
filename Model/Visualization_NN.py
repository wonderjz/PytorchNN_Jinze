#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:33:48 2024

@author: lijinze
"""


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torchviz import make_dot
from torchvision.models import vgg16  # 以 vgg16 为例


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        
        # Try to modify this DNN to achieve better performance
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            #nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Loss function MSE
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # You may try regularization here
        
        return self.criterion(pred, target)
    

'''
x = torch.randn(4, 3, 32, 32)  # 随机生成一个张量
model = vgg16()  # 实例化 vgg16，网络可以改成自己的网络
out = model(x)   # 将 x 输入网络
g = make_dot(out)  # 实例化 make_dot
g.view()  # 直接在当前路径下保存 pdf 并打开
# g.render(filename='netStructure/myNetModel', view=False, format='pdf')  # 保存 pdf 到指定路径不打开
'''

x = torch.randn(100,92)  # 随机生成一个张量
model = NeuralNet(input_dim=92)
#model = MYNeuralNet()
out = model(x)   # 将 x 输入网络
g = make_dot(out)  # 实例化 make_dot
g.view()  # 直接在当前路径下保存 pdf 并打开















