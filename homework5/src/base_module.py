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
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import cycle, islice

# plt.ion()

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
    sampled_points = points[samples_index[0],samples_index[1],:].contiguous().view(B,-1,C)
    remain_points = points[remains_index[0],remains_index[1],:].contiguous().view(B,-1,C) #(B,N-1,C)
    colors = np.array(["#FF0000","#000000"])

    for i in range(sample_nums-1): 
        dist = torch.sum(square_distance(remain_points,sampled_points), 2) #(B,N-i-1)
        max_index = torch.argmax(dist, 1) # (B)
        centriod = remain_points[index_B,max_index,:].view(B,-1,C) #(B,1,C)
        sampled_points = torch.cat((sampled_points,centriod),dim=1)  #(B,i+2,C)
        remain_points_index = torch.arange(remain_points.shape[1],dtype=torch.long).repeat(B,1) #(B,N-i-1)
        farthest_point_index = max_index.view(B,1).repeat(1,remain_points.shape[1]) #(B,N-i-1)
        mask = (remain_points_index!=farthest_point_index) #(B,N-i-1)
        remain_points = remain_points[mask].contiguous().view(B,-1,C) #(B,N-i-2,C)
        if plt_show:
            ax = plt.figure().add_subplot(111, projection = '3d')
            ax.scatter(sampled_points[1,:,0],sampled_points[1,:,1],sampled_points[1,:,2], s=2, color=colors[0])
            ax.scatter(remain_points[1,:,0],remain_points[1,:,1],remain_points[1,:,2], s=2, color=colors[-1])
            plt.show() 
    return sampled_points

def query_ball_points(radius, group_points_num, points, query_points):
    '''
    @Functions: Query points' neighbors
    @Args:
        radius:(float)
        group_points_num:(int) sample number, M
        points:(Tensor) all points, [B,N,C]
        query_points:(Tensor) query points, [B,S,C] 
    '''
    device = points.device
    B,N,C = points.shape
    _,S,_ = query_points.shape
    group_idx = torch.arange(N).view(1,1,N).to(device).repeat(B,S,1)
    dist = square_distance(query_points,points) # [B,S,N]
    group_idx[dist> radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:group_points_num]
    # if number of neighbors < group_points_num, instead of most nearest point
    group_first = group_idx[:,:,0].view(B,S,1).repeat(1,1,group_points_num)
    mask = group_idx==N
    group_idx[mask] = group_first[mask] #[B,S,group_points_num]
    return group_idx

def index_points(points, idx):
    '''
    @Functions: return point position data according idx
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
    batch_idx = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def sample_and_group(n_points, radius, n_samples, points, points_feature):
    '''
    @Functions: 
    @Args:
        n_points: Numbers of downsample centers
        n_samples: Numbers of neighbors per center
        points: point 
        points_feature: last layer's points feature
    @Return:
        fps_sampled_points: [B, n_points, n_sample, C]
        points_feature: [B, n_points, n_sample, C+D]
    '''
    B,N,C = points.shape
    S = n_points
    fps_sampled_points = farthest_point_sample(points, n_points)
    # Group points near the sample points acquire by FPS 
    idx = query_ball_points(radius, n_samples, points, fps_sampled_points)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - fps_sampled_points.view(B,S,1,C)

    if points_feature is not None:
        last_layer_feature = index_points(points_feature, idx)
        output_points = torch.cat((grouped_points_norm, last_layer_feature),dim=-1) # [B,S,sample,C+D] 
    else:
        output_points = grouped_points_norm    
    return fps_sampled_points, output_points

def sample_group_all(points_position, points_feature):
    '''
    @Functions: Take all points to group
    @Args:
        points_position:(Tensor) input position data [B,N,3]
        points_feature:(Tensor) input points data [B,N,D]
    @Return:
        points_sample_position: [B,1,3]
        grouped_points: [B,1,N,3+D]
    '''
    device = points_position.device
    B,N,C = points_position.shape
    # Take [0 ...] to be a center
    points_sample_position = torch.zeros(B, 1, C).to(device)
    grouped_points = points_position.view(B, 1, N, C)
    if points_feature is not None:
        grouped_points = torch.cat((grouped_points,points_feature.view(B,1,N,-1)),dim=-1)
    return points_sample_position,grouped_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_points, radius, n_samples, in_channels, mlp, group_all):
        super(PointNetSetAbstraction,self).__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_samples = n_samples
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bn = nn.ModuleList()
        for out_channels in mlp:
            self.mlp_convs.append(nn.Conv2d(in_channels,out_channels,1))
            self.mlp_bn.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
    def forward(self, points_position, points_feature):
        points_position = points_position.permute(0, 2, 1) # [B,C,N] -> [B,N,C]
        if points_feature is not None:
            points_feature = points_feature.permute(0, 2, 1)
        if self.group_all:
            sampled_points_position, sampled_points_feature = sample_group_all(points_position, points_feature)
        else:
            sampled_points_position, sampled_points_feature = sample_and_group(points_position, points_feature)
        sampled_points_feature = sampled_points_feature.permute(0,3,2,1)
        for i,conv in enumerate(self.mlp_convs):
            bn = self.mlp_bn[i]
            sampled_points_feature = F.relu(bn(conv(sampled_points_feature)), inplace=True)
        sampled_points_feature = torch.max(sampled_points_feature,2)[0]
        sampled_points_position = sampled_points_position.permute(0,2,1)
        return sampled_points_position,sampled_points_feature

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, n_points, radius_list, n_samples_list, in_channels, mlp_list):
        super(PointNetSetAbstractionMsg,self).__init__()
        self.n_points = n_points
        self.radius_list = radius_list
        self.n_samples_list = n_samples_list
        self.mlp_convs_block = nn.ModuleList()
        self.mlp_bn_block = nn.ModuleList()
        for mlp in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channels + 3
            for out_channels in mlp_list[mlp]:
                convs.append(nn.Conv2d(last_channel,out_channels,1))
                bns.append(nn.BatchNorm2d(out_channels))
                last_channel = out_channels
            self.mlp_convs_block.append(convs)
            self.mlp_bn_block.append(bns)

    def forward(self, points_position, points_feature):
        points_position = points_position.permute(0, 2, 1) # [B,C,N] -> [B,N,C]
        if points_feature is not None:
            points_feature = points_feature.permute(0, 2, 1)
        B,N,C = points_position.shape
        S = self.n_points
        downsampled_points = farthest_point_sample(points_position, S)
        features_list=[]
        for i,radius in enumerate(self.radius_list):
            # Sample and group points for different radius
            K = self.n_samples_list[i]
            group_index = query_ball_points(radius, K, points_position, downsampled_points)
            group_points = index_points(points_position, group_index)
            group_points -= downsampled_points.view(B,S,1,C)
            if points_feature is not None:
                group_points_feature = index_points(points_feature, group_index)
                group_points = torch.cat([group_points_feature, group_points],dim=-1)
            
            group_points = group_points.permute(0,3,2,1)
            for j in range(len(self.mlp_convs_block[i])):
                conv = self.mlp_convs_block[i][j]
                bn = self.mlp_bn_block[i][j]
                group_points = F.relu(bn(conv(group_points)), inplace=True)
            sampled_points_feature = torch.max(group_points,2)[0]
            features_list.append(sampled_points_feature)
        
        sampled_points_position = downsampled_points.permute(0,2,1)
        sampled_points_feature = torch.cat(features_list, dim=1)
        return sampled_points_position,sampled_points_feature

class PointNetFeaturePropagation(nn.Module):
    '''
    @Functions: Upsample points data for segment branch.
    '''
    def __init__(self, in_channels, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channels in mlp:
            self.mlp_convs.append(nn.Conv2d(in_channels, out_channels),1)
            self.mlp_bns.append(nn.BatchNorm1d(out_channels))
        
    def forward(self, points_position, sampled_points_position, skip_points_feature, sampled_points_feature):
        '''
        @Args:
            points_position:(Tensor) [B, C, N]
            sampled_points_position:(Tensor) [B, C, S]
            skip_points_feature:(Tensor) [B, D, N]
            sampled_points_feature:(Tensor) [B, D, S]
        @Return:
        '''
        
        points_position = points_position.permute(0,2,1)
        sampled_points_position = sampled_points_position.permute(0,2,1)

        sampled_points_feature = sampled_points_feature.permute(0,2,1)
        B,N,C = points_position.shape
        _,S,_ = sampled_points_position.shape

        if S==1:
            interpolated_points = sampled_points_feature.repeat(1,N,1)
        else:
            # Interpolate points from 3 nearest points
            dists = square_distance(points_position, sampled_points_position)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:,:,:3], idx[:,:,:3]

            # weight average
            weights = 1.0/(dists + 1e-8)
            norm = torch.sum(weights, dim=2, keepdim=True)
            weights = weights/norm
            interpolated_points = torch.sum(index_points(sampled_points_position,idx)*weights.view(B,N,3,1))

        if points_feature is not None:
            points_feature = points_feature.permute(0,2,1)
            points_feature = torch.cat([points_feature, interpolated_points], dim=-1)
        else:
            points_feature = interpolated_points
        
        points_feature = points_feature.permute(0,2,1)
        for i,conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points_feature = F.relu(bn(conv(points_feature)), inplace=True)
        return points_feature

if __name__ =="__main__":
    points = []
    for i in range(20):
        pc = np.random.randint(0,100,(40,3))
        points.append(pc)
    torch_pc = torch.from_numpy(np.array(points))
    # print("pc_normalize:",pc_normalize(pc))
    # print("pc_normalize_other:",pc_normalize_other(pc))
    # sample_pc = farthest_point_sample(torch_pc,10)
    
    print("Samples shape: ",sample_pc.shape)