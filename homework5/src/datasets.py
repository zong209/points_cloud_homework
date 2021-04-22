#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      datasets.py
@Time:      2021/04/15 15:24:50
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  数据准备
'''

import os
import math
import numpy as np
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, ratio=0.8, random=True, train_mode=True ,data_argument=True, jitter=True):
        '''
        @Args:
             root_dir:(String) root path to data dictory
             ratio:(float) ratio of train samples
             random:(bool) random data list
             train_mode:(bool) train model or val model
        '''
        self.root_dir = root_dir
        self.ratio = ratio
        self.random = random
        self.data_list = self._get_file_list()
        self.train_index, self.val_index = self._split_dataset()
        self.train_mode = train_mode
        self.data_argument = data_argument
        self.jitter = jitter
    
    def _get_file_list(self):
        '''
        @Functions: Get file path of ModelNet40
        '''
        datas_list = []
        self.label_to_index = {}
        self.index_to_label = []
        root_files = os.listdir(self.root_dir)
        for index, label in enumerate(root_files):
            self.label_to_index[label] = index
            self.index_to_label.append(label)
        self.kinds_path = [os.path.join(self.root_dir,filename) for filename in root_files if os.path.isdir(os.path.join(self.root_dir,filename))]
        for dictory in self.kinds_path:
            datas_list += [ os.path.join(dictory,file) for file in os.listdir(dictory) if file.endswith(".txt")]
        return datas_list

    def _split_dataset(self):
        '''
        @Functions: Split data list to train list and valid list
        @Return:    Index list
        '''
        datas_list = self.data_list
        length = len(datas_list)
        datas_array = np.array(datas_list)
        if self.random:
            random_index = np.random.choice(length, length ,replace=False)
        else:
            random_index = np.arange(length)
        split_boundary = math.floor(length*self.ratio)
        train_index = random_index[0:split_boundary]
        val_index = random_index[split_boundary::]
        return train_index, val_index

    def _load_file_data(self, file):
        '''
        @Functions: Load point data from txt
        '''
        assert os.path.exists(file), "{} is not exist".format(file)
        data = []
        with open(file,"r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                line = line.strip().split(",")
                if not len(line):
                    return data
                line_data = np.array([float(el) for el in line])
                data.append(line_data)
        return data

    def __getitem__(self, index):
        if self.train_mode:
            file = self.data_list[self.train_index[index]]
        else:
            file = self.data_list[self.val_index[index]]
        file_data = np.array(self._load_file_data(file))
        point_set = file_data[:,0:3]
        if self.data_argument:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            point_set[:,[0,1]] = np.dot(point_set[:,[0,1]],rotation_matix)
        if self.jitter:
            point_set += np.random.normal(0,0.02,size=point_set.shape)
        label = file.strip().split("/")[-2]
        label = self.label_to_index[label]
        return point_set, label
        
    def __len__(self):
        if self.train_mode:
            return len(self.train_index)
        else:
            return len(self.val_index)

    @property
    def label_dict(self):
        return self.label_to_index

if __name__=="__main__":

    dataset = ModelNet40Dataset(root_dir="/home/260243/gz/2021/points_cloud_homework/dataset/modelnet40_normal_resampled")
    print("Samples: ",len(dataset))
    data, label = dataset[0]
    print("Sample: ", data.shape,label)