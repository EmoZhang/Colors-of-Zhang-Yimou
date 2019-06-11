import os
from PIL import Image
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation
import colorsys
# %matplotlib inline


def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r = int(rgb[0])
    g = int(rgb[1])
    b = int(rgb[2])
    return '#%02x%02x%02x' % (r, g, b)


def pic(path):
    # %% Get color list
    im = Image.open(path)
    w, h = im.size
    colors = im.getcolors(w * h)

    # %% Check that the sum of colors match the number of pixels
    assert sum([colors[i][0] for i in range(len(colors))]) == w * h

    # %% Adjust color list
    for idx, item in enumerate(colors):

        count, (r, g, b) = item

        # 忽略纯黑色
        if not ((r > 200) & (g > 200) & (b > 200)):
            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            # 忽略高亮色
            if y <= 0.9:
                continue
            else:
                del colors[idx]
        else:
            del colors[idx]

    # %% Get DataFrame
    df = pd.DataFrame(
        data={
            'pixels': [colors[i][0] for i in range(len(colors))],
            'R': [colors[i][1][0] for i in range(len(colors))],
            'G': [colors[i][1][1] for i in range(len(colors))],
            'B': [colors[i][1][2] for i in range(len(colors))],
            # 'alpha': [colors[i][1][3] for i in range(len(colors))],
            'hex': [hexencode(colors[i][1]) for i in range(len(colors))]
        })

    # %%

    #
    hsv_matrix = np.zeros((len(df), 3))

    for i in range(len(df)):
        hsv_matrix[i] = colorsys.rgb_to_hsv(r=df.R[i] / 255, g=df.G[i] / 255, b=df.B[i] / 255)

    df['h'] = hsv_matrix[:, 0]
    df['s'] = hsv_matrix[:, 1]
    df['v'] = hsv_matrix[:, 2]

    dfHSV2 = df.copy()

    dfHSV2['hx'] = dfHSV2.s * np.sin(dfHSV2.h * 2 * np.pi)
    dfHSV2['hy'] = dfHSV2.s * np.cos(dfHSV2.h * 2 * np.pi)

    kmeansHSV = KMeans(n_clusters=4, random_state=0, n_init=10).fit(df[['h', 's', 'v']])
    kmeansHSV2 = KMeans(n_clusters=4, random_state=0, n_init=10).fit(dfHSV2[['hx', 'hy', 'v']])

    dfHSV2['kcenter'] = kmeansHSV2.labels_

    HSVcenters2 = np.zeros((kmeansHSV2.n_clusters, 3))
    for i in range(kmeansHSV2.n_clusters):
        HSVcenters2[i, :] = colorsys.hsv_to_rgb(h=kmeansHSV2.cluster_centers_[i, 0],
                                                s=kmeansHSV2.cluster_centers_[i, 1],
                                                v=kmeansHSV2.cluster_centers_[i, 2])
    HSVcenters2 *= 255

    weighted_cluster_centers2 = np.zeros((kmeansHSV2.n_clusters, 3))
    for c in range(kmeansHSV.n_clusters):
        weighted_cluster_centers2[c] = np.average(dfHSV2[dfHSV2['kcenter'] == c][['h', 's', 'v']],
                                                  weights=dfHSV2[dfHSV2['kcenter'] == c]['pixels'], axis=0)

    weighted_cluster_center_colors2 = np.zeros((kmeansHSV2.n_clusters, 3))
    for i in range(kmeansHSV2.n_clusters):
        weighted_cluster_center_colors2[i] = colorsys.hsv_to_rgb(
            h=weighted_cluster_centers2[i, 0],
            s=weighted_cluster_centers2[i, 1],
            v=weighted_cluster_centers2[i, 2])
    weighted_cluster_center_colors2 *= 255

    return path, weighted_cluster_centers2, weighted_cluster_center_colors2, kmeansHSV2


def save(idx, path, weighted_cluster_centers2, weighted_cluster_center_colors2, kmeansHSV2):
    #%% save to file
    decimals = 3  # less precision, and faster loading for browser plotting

    #data_array = pd.DataFrame(data={
    #    'x': np.round(dfHSV2.v * dfHSV2.s * np.sin(dfHSV2.h * 2 * np.pi), decimals),
    #    'y': np.round(dfHSV2.v * dfHSV2.s * np.cos(dfHSV2.h * 2 * np.pi), decimals),
    #    'z': np.round(dfHSV2.v, decimals),
    #    'c': dfHSV2.hex
    #})
    #data_array.to_csv('/Users/mac/Documents/machine-learning-notebooks-master/results/cluster-data.csv', index=False)

    x = weighted_cluster_centers2[:, 2] * weighted_cluster_centers2[:, 1] * np.sin(
        weighted_cluster_centers2[:, 0] * 2 * np.pi)
    y = weighted_cluster_centers2[:, 2] * weighted_cluster_centers2[:, 1] * np.cos(
        weighted_cluster_centers2[:, 0] * 2 * np.pi)
    z = weighted_cluster_centers2[:, 2]
    c = [hexencode(weighted_cluster_center_colors2[i]) for i in range(kmeansHSV2.n_clusters)]

    dominant_array = pd.DataFrame(data={
        'x2': np.round(x, decimals),
        'y2': np.round(y, decimals),
        'z2': np.round(z, decimals),
        'c2': c
    })
    dominant_array.to_csv(path+str(idx)+'dominant-data.csv',
                          index=False)

    # Combine and save
    #data_array.join(dominant_array).to_csv(
    #    '/Users/mac/Documents/machine-learning-notebooks-master/results/combined.csv', index=False)


def main():
    path = "/Users/mac/Downloads/frametest/test/"

    # 获取文件夹下所有PNG文件名
    files_raw = os.listdir(path)
    files = []
    for file in files_raw:
        if os.path.splitext(file)[1] == ".png":
            files.append(file)

    for idx, file in enumerate(files):
        image = path + file
        data = pic(image)
        save(idx, data[0], data[1], data[2], data[3])


if __name__ == '__main__':
    main()
