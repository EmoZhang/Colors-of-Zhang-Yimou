from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


# 读取文件
file = '/Users/mac/Downloads/frametest/Red.Sorghum/results/dominant_array.csv'
# 保存路径，自动加上/results/
path = '/Users/mac/Downloads/frametest/Red.Sorghum/'

df = pd.read_csv(file, header=0, sep=',')


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


area = 10
plotting(path, d_df, t='cylinder', p='c', w='count', s=10)
plotting(path, d_df, t='cylinder', p='c', w='percentage', s=10)
plotting(path, d_df, t='cone', p='c', w='count', s=10)
plotting(path, d_df, t='cone', p='c', w='percentage', s=10)
