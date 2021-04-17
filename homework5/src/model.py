#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      model.py
@Time:      2021/04/16 09:04:38
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  build pointNet++ Model
'''

import torch
import torch.nn as nn
import torch.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import cycle, islice

# plt.ion()6

def pc_normalize(pc):
    # Center Normalize
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.linalg.norm(pc,axis=1))
    return pc/m

def square_distance(src, dst):
    '''
    @Functions: Caculate Euclid distance of two points

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    @Args:
        src:(Tensor) [B,N,C]
        dst:(Tensor) [B,M,C]
    @Return:
        dist:(Tensor) [B,N,M]
    '''
    B,N,_ = src.shape
    _,M,_ = dst.shape
    dist = -2*torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).view(B,N,1)
    dist += torch.sum(dst**2, -1).view(B,1,M)
    return dist

def farthest_point_sample(points, sample_nums, plt_show = False):
    '''
    @Functions: FPS downsample
    @Args:
        points:(Tensor) [B,N,C]
        sample_nums:(int) desired number of points
    '''
    device = points.device
    B,N,C = points.shape

    # Split two points set
    index_B = torch.arange(B,dtype=torch.long).to(device)
    origin_index = torch.arange(N,dtype=torch.long).repeat(B,1).to(device)
    farthest = torch.randint(0,N,(B,1),dtype=torch.long).repeat(1,N).to(device)
    samples_index = torch.where(origin_index == farthest)
    remains_index = torch.where(origin_index != farthest)
    sample_points = points[samples_index[0],samples_index[1],:].contiguous().view(B,-1,C)
    remain_points = points[remains_index[0],remains_index[1],:].contiguous().view(B,-1,C) #(B,N-1,C)
    colors = np.array(["#FF0000","#000000"])

    for i in range(sample_nums-1): 
        ax = plt.figure().add_subplot(111, projection = '3d')
        dist = torch.sum(square_distance(remain_points,sample_points), 2) #(B,N-i-1)
        max_index = torch.argmax(dist, 1) # (B)
        centriod = remain_points[index_B,max_index,:].view(B,-1,C) #(B,1,C)
        sample_points = torch.cat((sample_points,centriod),dim=1)  #(B,i+2,C)
        remain_points_index = torch.arange(remain_points.shape[1],dtype=torch.long).repeat(B,1) #(B,N-i-1)
        farthest_point_index = max_index.view(B,1).repeat(1,remain_points.shape[1]) #(B,N-i-1)
        mask = (remain_points_index!=farthest_point_index) #(B,N-i-1)
        remain_points = remain_points[mask].contiguous().view(B,-1,C) #(B,N-i-2,C)
        if plt_show:
            ax.scatter(sample_points[1,:,0],sample_points[1,:,1],sample_points[1,:,2], s=2, color=colors[0])
            ax.scatter(remain_points[1,:,0],remain_points[1,:,1],remain_points[1,:,2], s=2, color=colors[-1])
            plt.show() 
    return sample_points

def query_ball_points(radius, group_points_num, points, query_points):
    '''
    @Functions: Query points' neighbors
    @Args:
        radius:(float)
        group_points_num:(int) sample number, M
        points:(Tensor) all points, [B,N,C]
        query_points:(Tensor) query points, [B,S,C] 
    '''
    B,N,C = points.shape
    _,S,_ = query_points.shape
    group_idx = torch.arange(N).view(1,1,N).to(device).repeat(B,S,1)
    dist = square_distance(query_points,points) # [B,S,N]
    group_idx[dist> radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:group_points_num]
    # if number of neighbors < group_points_num, instead of most nearest point
    group_first = group_idx[:,:,0].view(B,S,1).repeat(1,1,group_points)
    mask = group_idx==N
    group_idx[mask] = group_idx[mask] #[B,S,group_points_num]
    return group_idx

def index_points(points, idx):
    '''
    @Functions: return point value according idx
    @Args:
        points:(Tensor) [B,N,C]
        idx:(Tensor) [B,S]
    @Return: [B,S,C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] *(len(view_shape) -1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = np.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]



def sample_and_group(n_points, radius, n_samples, points, points_feature):
    '''
    @Functions: 
    @Args:
        points: point info
        points_feature: last layer's points feature
    @Return:
    
    '''
    B,N,C = points.shape
    S = n_points
    fps_sample_points = farthest_point_sample(points, n_points)
    # Group points near the sample points acquire by FPS 
    idx = query_ball_points(radius, n_samples, points, fps_sample_points)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - fps_sample_points.view(B,S,1,C)

    if points_feature not None:
        last_layer_feature = index_points(points_feature, idx)
        new_points = torch.cat((grouped_points_norm, last_layer_feature),dim=-1) # [B,S,S,C+D] 







# class PointNetSetAbstraction(nn.Module):
#     super(PointNetSetAbstraction,self).__init__()
#     def __init__(self):


if __name__ =="__main__":
    points = []
    for i in range(20):
        pc = np.random.randint(0,100,(40,3))
        points.append(pc)
    torch_pc = torch.from_numpy(np.array(points))
    # print("pc_normalize:",pc_normalize(pc))
    # print("pc_normalize_other:",pc_normalize_other(pc))
    sample_pc = farthest_point_sample(torch_pc,10)
    print("Samples shape: ",sample_pc.shape)