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

def ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, ratio=0.8, random=True):
        self.root_dir = root_dir
        self.ratio = ratio
        self.random = random
        self.data_list = self._get_file_list()
    
    def _get_file_list(self):
        datas_list = []
        root_files = os.listdir(self.root_dir)
        self.kinds_path = [filename for filename in root_files if os.path.isdir(filename)]
        for dictory in self.kinds_path:
            datas_list += os.listdir(dictory)
        return datas_list

    def _split_dataset(self):
        datas_list = self.data_list
        length = len(datas_list)
        datas_array = np.array(datas_list)
        random_index = np.random.choice(length, length ,replace=False)
        split_boundary = math.floor(random_index*self.random)
        train_index = random_index[0:split_boundary]
        val_index = random_index[split_boundary::]
        return datas_array[train_index],datas_array[val_index]

    def __len__(self):
        files_list = os.listdir(self.root_dir)
        self.bin_list = [filename for filename in files_list if filename.endswith(".bin")]
        return len(self.bin_list)

    def _get_

    def __getitem__(self, index):
        # 单个样本
        return

if __name__=="__main__":

    dataset = ModelNet40Dataset()