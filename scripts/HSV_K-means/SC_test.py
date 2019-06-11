from datetime import datetime

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


X, labels_true = make_blobs(n_samples=12000, n_features=3, centers=8,
                            cluster_std=[0.1, 0.05, 0.05, 0.1, 0.75, 0.75, 0.1, 0.1],
                            center_box=(-1.0, 1.0), shuffle=True, random_state=0)
t_s = datetime.now()


score_list = []  # 用来存储每个K下模型的平局轮廓系数
silhouette_int = -1  # 初始化的平均轮廓系数阀值
for n_clusters in range(3, 10):  # 遍历从2到10几个有限组
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # 建立聚类模型对象
    cluster_labels_tmp = model_kmeans.fit_predict(X)  # 训练聚类模型
    silhouette_tmp = metrics.silhouette_score(X, cluster_labels_tmp)  # 得到每个K下的平均轮廓系数
    score_list.append([n_clusters, silhouette_tmp])  # 将每次K及其得分追加到列表
    if silhouette_tmp > silhouette_int:  # 如果平均轮廓系数更高
        best_k = n_clusters  # 将最好的K存储下来
        silhouette_int = silhouette_tmp  # 将最好的平均轮廓得分存储下来
        # best_kmeans = model_kmeans  # 将最好的模型存储下来
        # cluster_labels_k = cluster_labels_tmp  # 将最好的聚类标签存储下来
print('{:*^60}'.format('K value and silhouette summary:'))
print(np.array(score_list))  # 打印输出所有K下的详细得分
print('Best K is:{0} with average silhouette of{1}'.format(best_k, silhouette_int.round(4)))


t_e = datetime.now()
usedtime = t_e - t_s
print('[%s]' % usedtime)
