import os
import colorsys

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Methods


def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r = int(rgb[0])
    g = int(rgb[1])
    b = int(rgb[2])
    return '#%02x%02x%02x' % (r, g, b)


def save_data(path, array, array_name):
    """输入路径和DataFrame，保存为csv文件"""
    array.to_csv(path + array_name + '.csv', index=False)


def kmeans_hsv(df, method=None, k=1):
    """输入一个包含HSV信息的DataFrame，可选不同方法估计K值或指定k值，再对其进行K-means聚类，默认k = 1，输出四个关键DataFrame"""

    # 增加极坐标hx, hy
    df['hx'] = df.s * np.sin(df.h * 2 * np.pi)
    df['hy'] = df.s * np.cos(df.h * 2 * np.pi)

    def kmeans_session(df, k):
        # # 对df[['h', 's', 'v']]kmeans
        # kmeansHSV = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df[['h', 's', 'v']])
        # 对df[['hx', 'hy', 'v']]kmeans
        kmeansHSV2 = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df[['hx', 'hy', 'v']])

        # 为df添加kcenter列，内容为kmeansHSV2的labels
        df['kcenter'] = kmeansHSV2.labels_

        # weighted_cluster_centers2空矩阵，k行3列
        weighted_cluster_centers2 = np.zeros((kmeansHSV2.n_clusters, 3))
        # 选择df中kcenter为c的行，以['pixels']为权重求['h', 's', 'v']每列均值，填入第c行
        # for c in range(kmeansHSV.n_clusters):
        for c in range(kmeansHSV2.n_clusters):
            weighted_cluster_centers2[c] = np.average(df[df['kcenter'] == c][['h', 's', 'v']],
                                                      weights=df[df['kcenter'] == c]['pixels'], axis=0)

        # weighted_cluster_center_colors2空矩阵，k行3列
        weighted_cluster_center_colors2 = np.zeros((kmeansHSV2.n_clusters, 3))
        # 把kmeansHSV2的k个中心分别填入weighted_cluster_center_colors2的每行
        for i in range(kmeansHSV2.n_clusters):
            weighted_cluster_center_colors2[i] = colorsys.hsv_to_rgb(
                h=weighted_cluster_centers2[i, 0],
                s=weighted_cluster_centers2[i, 1],
                v=weighted_cluster_centers2[i, 2])
        weighted_cluster_center_colors2 *= 255

        return df, kmeansHSV2, weighted_cluster_centers2, weighted_cluster_center_colors2

    if method == 'GS':
        k = Methods.gs(df[['hx', 'hy', 'v']].values)
        print('Gap Statistic: the best k is', k)
    elif method == 'SC':
        k = Methods.sc(df[['hx', 'hy', 'v']].values)
        print('Silhouette Coefficient: the best k is ', k)
    elif method == 'CH':
        k = Methods.ch(df[['hx', 'hy', 'v']].values)
        print('Calinski-Harabasz Index: the best k is ', k)
    else:
        print('The chosen k is', k)

    kmeans_result = kmeans_session(df, k)

    return kmeans_result


def load_pics(path):
    """输入路径，输出路径下所有PNG文件名列表"""

    # 获取文件夹下所有PNG文件名，并输出绝对路径
    files_raw = os.listdir(path)

    files = []

    for file in files_raw:
        if os.path.splitext(file)[1] == ".png" or os.path.splitext(file)[1] == ".PNG":
            file = path + file
            files.append(file)

    return files


def get_data(path, files):
    """输入路径和图片列表，输出一个DataFrame，包含K-means结果，以HSV空间显示"""

    def get_rgb(path, file):
        """输入路径和文件名，输出一个包含RGB和HEX信息的DataFrame"""

        # 获取颜色列表
        im = Image.open(file)
        w, h = im.size
        colors = im.getcolors(w * h)

        # 调整颜色列表
        for idx, item in enumerate(colors):

            count, (r, g, b) = item

            # 忽略纯黑色
            if not ((r > 200) & (g > 200) & (b > 200)):
                y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
                y = (y - 16.0) / (235 - 16)
                # 忽略高亮色
                if y <= 0.9:
                    continue
                else:
                    del colors[idx]
            else:
                del colors[idx]

        # 获取DataFrame
        df = pd.DataFrame(
            data={
                'pixels': [colors[i][0] for i in range(len(colors))],
                'R': [colors[i][1][0] for i in range(len(colors))],
                'G': [colors[i][1][1] for i in range(len(colors))],
                'B': [colors[i][1][2] for i in range(len(colors))],
                # 'alpha': [colors[i][1][3] for i in range(len(colors))],
                'hex': [hexencode(colors[i][1]) for i in range(len(colors))]
            })

        return df

    def get_hsv(df):
        """输入包含RGB信息的DataFrame，输出包含HSV数据的DataFrame"""

        hsv_matrix = np.zeros((len(df), 3))

        for i in range(len(df)):
            hsv_matrix[i] = colorsys.rgb_to_hsv(r=df.R[i] / 255, g=df.G[i] / 255, b=df.B[i] / 255)

        df['h'] = hsv_matrix[:, 0]
        df['s'] = hsv_matrix[:, 1]
        df['v'] = hsv_matrix[:, 2]

        return df

    def sdominant_df(results):
        """输入索引和K-means结果，输出两个精确度不同的包含单张图片主题色的DataFrame"""

        decimals = 3

        c = [hexencode(results[3][i]) for i in range(results[1].n_clusters)]
        c_vector = np.array(c).reshape(-1, 1)
        centers_colors = np.concatenate((results[2].astype('str'), c_vector), axis=1)

        df_dict = {}
        for i in range(len(centers_colors)):
            df_dict[tuple(centers_colors[i])] = df_dict.get(tuple(centers_colors[i]), 0) + 1

        df_list = []
        keys = df_dict.keys()
        for key in keys:
            value = df_dict[key]
            df_list.append(np.append(str(value), np.array(key)))
        df_array = np.array(df_list)
        df_array_part = df_array[:, :4].astype('float')
        df_array_c = df_array[:, 4]

        pixels = df_array[:, 0].astype('int')
        x = df_array_part[:, 3] * df_array_part[:, 2] * np.sin(df_array_part[:, 1] * 2 * np.pi)
        y = df_array_part[:, 3] * df_array_part[:, 2] * np.cos(df_array_part[:, 1] * 2 * np.pi)
        z = df_array_part[:, 3]
        h = df_array_part[:, 1]
        s = df_array_part[:, 2]
        v = df_array_part[:, 3]
        c = df_array_c

        sdominant_array_raw_s = pd.DataFrame(data={
            'pixels': pixels,
            'x2': x,
            'y2': y,
            'z2': z,
            'c2': c,
            'h': h,
            's': s,
            'v': v
        })

        sdominant_array_s = pd.DataFrame(data={
            'x2': np.round(x, decimals),
            'y2': np.round(y, decimals),
            'z2': np.round(z, decimals),
            'c2': c,
        })

        return sdominant_array_raw_s, sdominant_array_s

    # 遍历文件列表进行K-means，输出结果到两个精确度不同的DataFrame中
    array_raw_list = []
    array_list = []

    for file in files:
        df = get_rgb(path, file)
        df = get_hsv(df)
        results = kmeans_hsv(df)
        dfs = sdominant_df(results)
        array_raw_list.append(dfs[0])
        array_list.append(dfs[1])

    sdominant_array_raw = pd.concat(array_raw_list)
    sdominant_array = pd.concat(array_list)

    # 把DataFrame保存为csv文件
    save_data(path, sdominant_array_raw, 'sdominant_array_raw')
    save_data(path, sdominant_array, 'sdominant_array')

    return sdominant_array_raw


def kmeans_final(df, path, method=None, k=1):

    def save_csv(path, results):

        decimals = 3  # less precision, and faster loading for browser plotting

        data_array = pd.DataFrame(data={
            'x': np.round(results[0].v * results[0].s * np.sin(results[0].h * 2 * np.pi), decimals),
            'y': np.round(results[0].v * results[0].s * np.cos(results[0].h * 2 * np.pi), decimals),
            'z': np.round(results[0].v, decimals),
            'c': results[0].hex
        })
        data_array.to_csv(path+'cluster-data.csv', index=False)

        x = results[2][:, 2] * results[2][:, 1] * np.sin(results[2][:, 0] * 2 * np.pi)
        y = results[2][:, 2] * results[2][:, 1] * np.cos(results[2][:, 0] * 2 * np.pi)
        z = results[2][:, 2]
        c = [hexencode(results[3][i]) for i in range(results[1].n_clusters)]

        dominant_array = pd.DataFrame(data={
            'x2': np.round(x, decimals),
            'y2': np.round(y, decimals),
            'z2': np.round(z, decimals),
            'c2': c
        })
        dominant_array.to_csv(path+'dominant-data.csv', index=False)

        # Combine and save
        data_array.join(dominant_array).to_csv(
            path+'combined.csv', index=False)

    def plotting(path, results):

        df, kmeansHSV2, weighted_cluster_centers2, weighted_cluster_center_colors2 = results

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        x = df.v * df.s * np.sin(df.h * 2 * np.pi)
        y = df.v * df.s * np.cos(df.h * 2 * np.pi)
        z = df.v
        c = df.hex
        s = df.pixels * 100

        ax.scatter(x, y, z, c=c, s=s, alpha=.7, edgecolor='k', lw=0)

        x = weighted_cluster_centers2[:, 2] * weighted_cluster_centers2[:, 1] * np.sin(
            weighted_cluster_centers2[:, 0] * 2 * np.pi)
        y = weighted_cluster_centers2[:, 2] * weighted_cluster_centers2[:, 1] * np.cos(
            weighted_cluster_centers2[:, 0] * 2 * np.pi)
        z = weighted_cluster_centers2[:, 2]
        c = [hexencode(weighted_cluster_center_colors2[i]) for i in range(kmeansHSV2.n_clusters)]

        ax.scatter(x, y, z, c=c, s=1000, alpha=1, edgecolor='k', lw=1)

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(0, 1)

        # ax.set_xlabel('H', fontsize=14)
        # ax.set_ylabel('S', fontsize=14)
        ax.set_zlabel('V', fontsize=14)

        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        ax.tick_params(
            axis='y',
            which='both',
            bottom=False,
            top=False,
            right=False,
            left=False,
            labelbottom=False,
            labelright=False,
            labelleft=False)

        plt.show()
        plt.savefig(path+'hsv-scatter-centers.png', bbox_inches='tight')

    df = df.rename(columns={'c2': 'hex'})
    results = kmeans_hsv(df, method=method, k=k)
    plotting(path, results)
    save_csv(path, results)
