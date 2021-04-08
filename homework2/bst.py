#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      bst.py
@Time:      2021/04/08 09:27:39
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  BST Algorithm
'''

import numpy as np

class TreeNode:
    def __init__(self, axis, value, index, left=None,right=None):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.index = index

class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index
    def __lr__(self, other):
        return self.distance < other.distance

class KnnResult:
    def __init__(self, capality, max_distance):
        self.capality= capality
        self.count = 0
        self.max_distance = max_distance
        self.dist_index = []
        for i in range(self.capality):
            self.dist_index.append(DistIndex(max_distance,0))
    
    def full(self):
        return self.count == self.capality
    
    def insert(self, dist, index):
        if dist >= self.max_distance:
            return
        if self.count < self.capality:
            self.count += 1
        i = self.count - 1
        while i>0:
            if dist < self.dist_index[i-1].distance:
                self.dist_index[i] = copy.deepcopy(self.dist_index[i-1])
                i-=1
            else:
                break
        self.dist_index[i] = DistIndex(dist,index)
        self.worst_value = self.dist_index[self.capality-1].distance

class BST:
    def __init__(self, nums, axis, worst_value=1e5):
        self.lens = len(nums)
        self.tree = self.build(nums, axis)
        self.worst_value = worst_value

    @staticmethod
    def build(nums, axis):
        def traverse(nums):
            if len(nums)<=0:
                return None
            mid_idx = len(nums)//2
            while mid_idx<len(nums)-1 and nums[mid_idx+1]==nums[mid_idx]:
                mid_idx += 1
            root = TreeNode(axis, nums[mid_idx], sort_index[mid_idx])
            root.left = traverse(nums[0:mid_idx])
            root.right = traverse(nums[mid_idx+1::])
            return root
        nums_array = np.array(nums)
        sort_index = list(np.argsort(nums_array))
        nums = list(nums_array[sort_index])
        return traverse(nums)

    def search(self, key, axis):
        root = self.tree
        while root:
            if root.value == key:
                return True
            elif key < root.value:
                root = root.left
            else:
                root = root.right
        return False
    

def knn_search_bst(points, result):
    
   

if __name__ == "__main__":
    nums=[9,4,2,5,1,7,1,4]
    bst = BST(nums,2)
    print(bst.search(5,0))