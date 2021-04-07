# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import math
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    point_cloud = np.array([point_cloud.x,point_cloud.y,point_cloud.z]).T
    min_boundary = np.min(point_cloud,axis=0)
    max_boundary = np.max(point_cloud,axis=0)
    Dx,Dy,Dz = np.ceil((max_boundary - min_boundary)/leaf_size)
    voxel_grid_nums = int(Dx*Dy*Dz)
    print("voxel_grid_nums",voxel_grid_nums)
    assert voxel_grid_nums<point_cloud.shape[0],"grid size({}<{}) is too small, please choose bigger one".format(point_cloud.shape[0],voxel_grid_nums)
    grids = [list([]) for _ in range(voxel_grid_nums)]
    for i in range(point_cloud.shape[0]):
        x,y,z = np.floor((point_cloud[i] -min_boundary)/leaf_size)
        grids[int(x+y*Dx+z*Dx*Dy)].append(point_cloud[i])
    for points in grids:
        if points:
            filtered_points.append(points[0])
        
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "dataset/MobileNet40_ply_points/chair/train/chair_0005.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # # 加载原始点云，txt处理
    # point_cloud_raw = np.genfromtxt(r"dataset/modelnet40_normal_resampled/bed/bed_0005.txt", delimiter=",")  #为 xyz的 N*3矩阵
    # point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    # point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    # point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 40)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
