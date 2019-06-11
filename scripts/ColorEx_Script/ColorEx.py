import os
import colorsys

from haishoku.haishoku import Haishoku
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Haishoku_plus


def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r = int(rgb[0])
    g = int(rgb[1])
    b = int(rgb[2])
    return '#%02x%02x%02x' % (r, g, b)


def load_pics(path):
    """输入路径，输出路径下所有PNG文件列表"""

    # 获取文件夹下所有PNG文件名，排序，并输出绝对路径
    files_raw = os.listdir(path)

    files = []

    for file in files_raw:
        if os.path.splitext(file)[1] == ".png" or os.path.splitext(file)[1] == ".PNG":
            file = path + file
            files.append(file)

    files.sort()

    return files


def get_data(path, files):
    """获取数据并保存到sesults文件夹"""

    def get_rgb(path, file, idx):
        """获取图片的颜色信息，保存主要颜色和调色板图片，返回元组/Dataframe"""

        # 判断是否存在results/colors文件夹，如果不存在则创建为文件夹
        folder = os.path.exists(path + 'results/colors')
        if not folder:
            os.makedirs(path + 'results/colors')

        # # 获取图片主要颜色元组(R,G,B)
        # dominant = Haishoku.getDominant(file)

        # 获取图片调色板列表[(percentage, (R,G,B)), ...]
        palette = Haishoku.getPalette(file)

        dominant = palette[0]

        # # 保存主要颜色图片
        Haishoku_plus.saveDominant(file, path, idx)

        # 保存调色板图片
        Haishoku_plus.savePalette(file, path, idx)

        # 转换为DataFrame

        df_list = []
        level_dict = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H'
        }

        for i, c in enumerate(palette):
            df_color = pd.DataFrame(
                data={
                    'pic': [idx],
                    'level': [level_dict[i]],
                    'percentage': [c[0]],
                    'r': [c[1][0]],
                    'g': [c[1][1]],
                    'b': [c[1][2]],
                    'hex': [hexencode(c[1])]
                })
            df_list.append(df_color)
        df_palette = pd.concat(df_list, ignore_index=True)

        return dominant, df_palette

    def get_hsv(df, index):
        """输入包含RGB数据的DataFrame，输出包含HSV数据和粗略柱坐标的DataFrame"""

        hsv_matrix = np.zeros((index, 3))

        for i in range(index):
            hsv_matrix[i] = colorsys.rgb_to_hsv(r=df.r[i] / 255, g=df.g[i] / 255, b=df.b[i] / 255)

        df['h'] = hsv_matrix[:, 0]
        df['s'] = hsv_matrix[:, 1]
        df['v'] = hsv_matrix[:, 2]

        # 保存较低精确度的柱坐标数据以便plotly使用
        decimals = 3

        y = df.s * np.sin(df.h * 2 * np.pi)
        x = df.s * np.cos(df.h * 2 * np.pi)
        z = df.v

        df['x'] = np.round(x, decimals)
        df['y'] = np.round(y, decimals)
        df['z'] = np.round(z, decimals)

        return df

    def save_data(path, df, df_name):
        """输入路径和DataFrame，保存为csv文件"""

        df.to_csv(path + 'results/' + df_name + '.csv', index=False)

    def plotting(path, df, t='cylinder', p='c', w='count', s=50):
        """在HSV柱坐标系中绘制散点图"""

        if t == 'cone':  # 锥
            cos = df.v * df.s * np.cos(df.h * 2 * np.pi)
            sin = df.v * df.s * np.sin(df.h * 2 * np.pi)
        else:  # 柱
            cos = df.s * np.cos(df.h * 2 * np.pi)
            sin = df.s * np.sin(df.h * 2 * np.pi)

        if p == 's':  # x对应sin
            x = sin
            y = cos
        else:  # x对应cos
            x = cos
            y = sin

        if w == 'percentage':
            weight = df.weight_percentage
        else:
            weight = df.weight_count

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # y = df.s * np.sin(df.h * 2 * np.pi)
        # x = df.s * np.cos(df.h * 2 * np.pi)
        z = df.v
        c = df.hex
        s = weight * s

        ax.scatter(x, y, z, c=c, s=s, alpha=.3, edgecolor='k', lw=0)

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(0, 1)

        # ax.set_xlabel('H', fontsize=14)
        # ax.set_ylabel('S', fontsize=14)
        # ax.set_zlabel('V', fontsize=14)

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

        plt.savefig(path + 'results/hsv-scatter-plot_' + "%s_%s" % (t, w) + '.png', bbox_inches='tight')

    # 判断是否存在results文件夹，如果不存在则创建为文件夹
    folder = os.path.exists(path + 'results')
    if not folder:
        os.makedirs(path + 'results')

    # 遍历所有图片，获得元组/DataFrame，一个只有dominant color，另一个包含所有palette
    df_palette_list = []

    d_wn_dict = {}
    d_wp_dict = {}

    files_length = len(files)
    for idx, file in enumerate(files):
        print('Loading data...', idx + 1, '/', files_length)
        dominant, df_palette = get_rgb(path, file, idx)
        df_palette_hsv = get_hsv(df_palette, len(df_palette))
        df_palette_list.append(df_palette_hsv)
        d_wn_dict[dominant[1]] = d_wn_dict.get(dominant[1], 0) + 1
        d_wp_dict[dominant[1]] = d_wp_dict.get(dominant[1], 0) + dominant[0]

    palette_array = pd.concat(df_palette_list, ignore_index=True)

    d_list = []
    keys = d_wn_dict.keys()
    for key in keys:
        wn = d_wn_dict[key]
        wp = d_wp_dict[key]
        d_list.append([wn] + [wp] + list(key))
    d_array = np.array(d_list)
    d_df = pd.DataFrame(d_array, None, ['weight_count', 'weight_percentage', 'r', 'g', 'b'])
    d_df = get_hsv(d_df, len(d_df))
    d_df['hex'] = [hexencode((d_df.r[i], d_df.g[i], d_df.b[i])) for i in range(len(d_df))]

    # 把DataFrame保存为csv文件
    save_data(path, d_df, 'dominant_array')
    save_data(path, palette_array, 'palette_array')

    # 绘制散点图
    plotting(path, d_df, t='cylinder', p='c', w='count', s=10)
    plotting(path, d_df, t='cylinder', p='c', w='percentage', s=10)
    plotting(path, d_df, t='cone', p='c', w='count', s=10)
    plotting(path, d_df, t='cone', p='c', w='percentage', s=10)
