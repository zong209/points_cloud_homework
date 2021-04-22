#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      pointnet_cls.py
@Time:      2021/04/19 16:08:49
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  pointsNet++ classifier
'''

import torch.nn as nn
import torch.nn.functional as F
from base_module import PointNetSetAbstraction, PointNetSetAbstractionMsg

class PointNetClassfier(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNetClassfier, self).__init__()
        in_channels = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4],[16,32,128], in_channels,[[32,32,64],[64,64,128],[64,96,128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2,0.4,0.8],[32,64,128], 320, [[64,64,128], [128,128,256],[128,128,256]])
        self.sa3 = PointNetSetAbstraction(None,None,None, 640+3, [256,512,1024],True)
        self.fc1 = nn.Linear(1024,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
    
    def forward(self, points):
        B,_,_ = points.shape
        if self.normal_channel:
            norm = points[:,3:,:]
            points = points[:,:3,:]
        else:
            norm = None
        sample_points1, feature1 = self.sa1(points, norm)
        sample_points2, feature2 = self.sa2(sample_points1,feature1)
        sample_points3, feature3 = self.sa3(sample_points2,feature2) 
        x = feature3.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        x = self.fc3(x)
        x = F.log_softmax(x,-1)
        return x, feature3

class PointNetClassfierLoss(nn.Module):
    def __init__(self):
        super(PointNetClassfierLoss,self).__init__()
    
    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss
