# 文件功能： 实现 K-Means 算法

import numpy as np

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.cluster_index = []

    def fit(self, data):
        diff_center = 1e5
        iter_nums = 0
        data_nums = data.shape[0]
        center = data[np.random.randint(0,data_nums,self.k_),:]
        while True:
            if diff_center<self.tolerance_ or iter_nums>self.max_iter_:
                return
            # Cacluate mean and varience
            distance_array = []
            new_center = []
            for i in range(self.k_):
                center_array = np.expand_dims(center[i,:],axis=0).repeat(data_nums,axis=0)
                distance = np.linalg.norm(data-center_array,axis=1)
                distance_array.append(distance)
            distance_array = np.array(distance_array)
            # Generate new center
            self.cluster_index = np.argsort(distance_array,axis=0)[0,:]
            for i in range(self.k_):
                cluster_i = data[np.where(self.cluster_index==i)]
                new_center.append(np.mean(cluster_i,axis=0))
            new_center = np.array(new_center)
            diff_center = np.sum(np.linalg.norm(new_center-center,axis=1))
            center = new_center
            iter_nums += 1

    def predict(self, p_datas):
        result = []
        result = [el for el in self.cluster_index]
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    cat = k_means.predict(x)
    print(max(cat))

