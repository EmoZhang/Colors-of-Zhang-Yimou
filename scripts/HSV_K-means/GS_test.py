from datetime import datetime
import scipy
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

dst = euclidean

X, labels_true = make_blobs(n_samples=12000, n_features=3, centers=8,
                            cluster_std=[0.1, 0.05, 0.05, 0.1, 0.75, 0.75, 0.1, 0.1],
                            center_box=(-1.0, 1.0), shuffle=True, random_state=0)

t_s = datetime.now()

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


def gap(data, refs=None, nrefs=20, ks=range(3, 10)):
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
            [dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index], :]) for current_row_index
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

    return ks, gaps, best_k


print(gap(X))

t_e = datetime.now()
usedtime = t_e - t_s
print('[%s]' % usedtime)
