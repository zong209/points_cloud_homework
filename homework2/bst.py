#! /usr/bin/env python
#-*- encoding:utf-8 -*-
'''
@File:      bst.py
@Time:      2021/04/08 09:27:39
@Author:    Gaozong/260243
@Contact:   260243@gree.com.cn/zong209@163.com
@Describe:  BST Algorithm
'''
import copy
import numpy as np

class TreeNode:
    def __init__(self, axis, value, index, left=None,right=None):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.index = index # maybe np.array or int
    
    @property
    def is_leaf(self):
        if self.left==None and self.right==None:
            return True
        else:
            return False

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
    
    @property
    def worst_dist(self):
        return self.dist_index[-1].distance
    
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
        # self.worst_value = self.dist_index[self.capality-1].distance

class BST:
    def __init__(self, nums, axis=0, worst_value=1e5):
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
    
    def knn_search(self, key, k):
        result = KnnResult(k, self.worst_value)
        root = self.tree
        while root:
            if result.worst_dist ==0:
                break
            result.insert(abs(root.value-key), root.index)
            if root.value >= key:
                root = root.left
            else:
                root = root.right
        return result

class KDTree:
    def __init__(self, points, leaf_size):
        assert len(points)>0, "Points can not be None"
        self.points = np.array(points)
        self.data_N = len(points)
        self.dimension = self.points.size/self.data_N
        self.leaf_size = leaf_size
        self.kd_tree = self.build_kd_tree()

    def change_axis(self, axis):
        axis+=1
        if axis==self.dimension:
            axis=0
        return axis

    def build_axis(self, index_list, axis=0):
        axis_points = self.points[index_list, axis]
        sort_index = np.argsort(axis_points)
        middle = index_list.shape[0]//2
        left_sort_index = sort_index[0:middle]
        right_sort_index = sort_index[middle::]
        axis = self.change_axis(axis)
        root = TreeNode(axis, axis_points[sort_index[middle]], index_list[sort_index[middle]])
        root.index = index_list
        if(index_list.shape[0]<=self.leaf_size):
            root.left=None
            root.right=None
        else:
            root.left = self.build_axis(index_list[left_sort_index], axis)
            root.right = self.build_axis(index_list[right_sort_index], axis)
        return root
    
    def build_kd_tree(self):
        init_index = np.array([i for i in range(self.data_N)])
        return self.build_axis(init_index, axis=0)
    
    def knn_search(self, k, key_point):
        root = self.kd_tree
        result = KnnResult(k, max_distance=1e5)
        axis = 0
        while root:
            if root.is_leaf:
                diff =  np.linalg.norm(np.expand_dims(key_point,0)- self.points[root.index,:],axis=1)
                for i in range(diff.shape[0]):
                    result.insert(diff[i], root.index[i])
            if key_point[axis] <= root.value:
                root = root.left
            else:
                root = root.right
            axis = self.change_axis(axis)
        return result

if __name__ == "__main__":
    # nums=[9,4,2,5,1,7,1,4]
    # bst = BST(nums)
    # result = bst.knn_search(3,3)
    # print([ nums[el.index] for el in result.dist_index])   

    db_size = 74
    dim = 3
    leaf_size = 4
    k = 3

    db_np = np.random.rand(db_size, dim)
    kd_tree = KDTree(db_np, leaf_size)
    result = kd_tree.knn_search(3, np.array([0,0,0]))
    reference = np.linalg.norm(np.expand_dims(np.array([0,0,0]),0)- db_np,axis=1)
    select_index = []
    distance_array = []
    for el in result.dist_index:
        select_index.append(el.index)
        distance_array.append(el.distance)
    print("K points:", db_np[select_index,:])
    print("Reference Distance:", reference)
    print("Select Distance:", distance_array) 