# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    sigma = 0.2
    num_iter = 1000
    pre_total = 0
    best_a,best_b,best_c,best_d = 0,0,0,0
    
    sz = data.shape[0]
    
    for index in range(num_iter):
        while True:
            sample_index = random.sample(range(sz),3)
            p = data[sample_index,:]
            if np.linalg.matrix_rank(p)==3:
                break
        v1 = p[2] - p[0]
        v2 = p[1] - p[0]
        
        cp = np.cross(v1,v2);
        a, b, c = cp
        d = np.dot(cp, p[2])
        
        dist = abs((a*data[:,0]+b*data[:,1]+c*data[:,2]-d)/(np.sqrt(a*a+b*b+c*c)))
        total_inliner = (dist<sigma).sum()
        
        if total_inliner>pre_total:
            pre_total = total_inliner
            best_a = a
            best_b = b
            best_c = c
            best_d = d

    print(best_a,best_b,best_c,best_d)
    dist = abs((a*data[:,0]+b*data[:,1]+c*data[:,2]-d)/(np.sqrt(a*a+b*b+c*c)))
    segmengted_cloud = data[dist>=sigma,:]
    
    # # # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    from sklearn.cluster import k_means
    kmeans = k_means(data,n_clusters=5)

    clusters_index = kmeans[1]
    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

# def main():
#     root_dir = 'data/' # 数据集路径
#     cat = os.listdir(root_dir)
#     cat = cat[1:]
#     iteration_num = len(cat)

#     for i in range(iteration_num):
#         filename = os.path.join(root_dir, cat[i])
#         print('clustering pointcloud file:', filename)

#         origin_points = read_velodyne_bin(filename)
#         segmented_points = ground_segmentation(data=origin_points)
#         cluster_index = clustering(segmented_points)

#         plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    origin_points = read_velodyne_bin("/home/260243/gz/2021/points_cloud_homework/homework4/C4/ransac/000000.bin")
    segmented_points = ground_segmentation(data=origin_points)
    cluster_index = clustering(segmented_points)
    plot_clusters(segmented_points, cluster_index)
    # main()
