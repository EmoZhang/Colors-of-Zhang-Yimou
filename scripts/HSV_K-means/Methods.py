import scipy
from scipy.spatial.distance import euclidean as dst
from sklearn import metrics
from sklearn.cluster import KMeans


def gs(X):
    """Gap Statistic"""

    KMeans_args_dict = {
        'n_clusters': 0,
        # drastically saves convergence time
        'init': 'k-means++',
        'random_state': 0,
        'max_iter': 300,
        'n_init': 10,
        'verbose': 0,
        # 'n_jobs':8
    }

    # def gap(data, refs=None, nrefs=20, ks=range(3, 10)):
    def gap(data, refs=None, nrefs=20, ks=range(1, 2)):
        """
        I: NumPy array, reference matrix, number of reference boxes, number of clusters to test
        O: Gaps NumPy array, Ks input list

        Give the list of k-values for which you want to compute the statistic in ks. By Gap Statistic
        from Tibshirani, Walther.
        """
        shape = data.shape

        if not refs:
            tops = data.max(axis=0)
            bottoms = data.min(axis=0)
            dists = scipy.matrix(scipy.diag(tops - bottoms))
            rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
            for i in range(nrefs):
                rands[:, :, i] = rands[:, :, i] * dists + bottoms
        else:
            rands = refs

        gaps = scipy.zeros((len(ks),))

        for (i, k) in enumerate(ks):
            KMeans_args_dict['n_clusters'] = k
            kmeans = KMeans(**KMeans_args_dict)
            kmeans.fit(data)
            (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_

            disp = sum(
                [dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index], :]) for
                 current_row_index
                 in range(shape[0])])

            refdisps = scipy.zeros((rands.shape[2],))

            for j in range(rands.shape[2]):
                kmeans = KMeans(**KMeans_args_dict)
                kmeans.fit(rands[:, :, j])
                (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_
                refdisps[j] = sum(
                    [dst(rands[current_row_index, :, j], cluster_centers[point_labels[current_row_index], :]) for
                     current_row_index in range(shape[0])])

            # let k be the index of the array 'gaps'
            gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)

        gaps = list(gaps)
        best_k = gaps.index(max(gaps)) + 1

        return best_k

    best_k = gap(X)

    return best_k


def sc(X):
    """Silhouette Coefficient"""

    global best_k
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

    return best_k


def ch(X):
    """Calinski-Harabasz Index"""

    global best_k
    score_list = []  # 用来存储每个K下模型的平局轮廓系数
    calinski_harabasz_int = 0  # 初始化的平均轮廓系数阀值
    for n_clusters in range(3, 10):  # 遍历从2到10几个有限组
        model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # 建立聚类模型对象
        cluster_labels_tmp = model_kmeans.fit(X)  # 训练聚类模型
        labels = cluster_labels_tmp.labels_
        # silhouette_tmp = metrics.silhouette_score(X, cluster_labels_tmp)  # 得到每个K下的平均轮廓系数
        calinski_harabasz_tmp = metrics.calinski_harabasz_score(X, labels)
        score_list.append([n_clusters, calinski_harabasz_tmp])  # 将每次K及其得分追加到列表
        if calinski_harabasz_tmp > calinski_harabasz_int:  # 如果平均轮廓系数更高
            best_k = n_clusters  # 将最好的K存储下来
            calinski_harabasz_int = calinski_harabasz_tmp  # 将最好的平均轮廓得分存储下来
            # best_kmeans = model_kmeans  # 将最好的模型存储下来
            # cluster_labels_k = cluster_labels_tmp  # 将最好的聚类标签存储下来

    return best_k
