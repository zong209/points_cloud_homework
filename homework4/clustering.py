# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
import math
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    # 使用RANSAC进行平面拟合
    p = 0.99 #能采样到好数据的概率
    e = 0.6  #离群点的占比
    s = 3    #至少3个点
    N = math.ceil(math.log(1-p)/math.log(1-(1-e)**s))

    iter_num = 0 #迭代次数
    max_inliers = 0  #拟合点数目
    inliers_points_bool = None
    dist_threshold = 0.2 #距离阈值
    best_a,best_b,best_c,best_d = 0,0,0,0
    while iter_num<N:
        #随机三个采样点
        points = data[np.random.randint(0, data.shape[0],3),:]
        #判断是否共线
        if np.linalg.matrix_rank(points) <3:
            continue
        line1 = points[1,:] - points[0,:]
        line2 = points[2,:] - points[0,:]
        #点法式计算平面参数
        params = np.cross(line1,line2)
        a, b, c = params
        d = np.dot(params,points[0,:])
        # 计算法向量与z(0,0,1)轴的夹角
        cos_angle = abs(np.dot(params,np.array([0,0,1]))/np.linalg.norm(params))
        if cos_angle < 0.707:#夹角如果大于45度
            continue
        # 计算所有点到平面的距离
        new_points = np.concatenate((data.T,np.ones((1,data.shape[0]))))
        all_params = np.array([a,b,c,d])
        dist_array = np.abs(np.dot(all_params,new_points)/np.linalg.norm(all_params))
        inliers_bool = dist_array<dist_threshold
        inliers = sum(inliers_bool)
        if inliers>max_inliers:
            max_inliers = inliers
            inliers_points_bool = inliers_bool
            best_a,best_b,best_c,best_d = a,b,c,d
        iter_num += 1
    print("Ground fit params:",best_a,best_b,best_c,best_d)
    segmengted_cloud = data[inliers_points_bool==False]
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


    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data):
# def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                          '#f781bf', '#a65628', '#984ea3',
    #                                          '#999999', '#e41a1c', '#dede00']),
    #                                   int(max(cluster_index) + 1))))
    # colors = np.append(colors, ["#000000"])
    colors = ['#377eb8']*data.shape[0]
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors)
    plt.show()

def main():
    root_dir = '/home/260243/gz/2021/points_cloud_homework/data' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[0:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        # cluster_index = clustering(segmented_points)

        plot_clusters(origin_points)

if __name__ == '__main__':
    main()
