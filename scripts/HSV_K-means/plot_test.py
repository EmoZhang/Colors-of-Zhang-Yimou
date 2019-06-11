from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation
import colorsys
# %matplotlib inline


def hexencode(rgb):
    """Transform an RGB tuple to a hex string (html color)"""
    r=int(rgb[0])
    g=int(rgb[1])
    b=int(rgb[2])
    return '#%02x%02x%02x' % (r,g,b)


# im = Image.open('img/bricks.png')
im = Image.open('/Users/mac/Documents/machine-learning-notebooks-master/img/bricks-scaled.png')
w, h = im.size
colors = im.getcolors(w*h)


#%% Check that the sum of colors match the number of pixels
assert sum([colors[i][0] for i in range(len(colors))]) == w*h


#%% Get DataFrame

df = pd.DataFrame(
    data={
        'pixels': [colors[i][0] for i in range(len(colors))],
        'R': [colors[i][1][0] for i in range(len(colors))],
        'G': [colors[i][1][1] for i in range(len(colors))],
        'B': [colors[i][1][2] for i in range(len(colors))],
        'alpha': [colors[i][1][3] for i in range(len(colors))],
        'hex': [hexencode(colors[i][1]) for i in range(len(colors))]
    })

#%% Plot individual colors

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = df.R
y = df.G
z = df.B
c = df.hex
s = df.pixels * 15

ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)

ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)
# plt.savefig('figures/rgb-scatter.png', bbox_inches='tight')
plt.show()

#%% RGB k-means
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(df[['R', 'G', 'B']])

df['kcenter'] = kmeans.labels_

# Calculate weighted average of RBG color

avg_col = np.zeros((kmeans.n_clusters, 3))
for c in range(kmeans.n_clusters):
    temp_df = df[df.kcenter == c]
    avg_col[c, 0] = np.average(temp_df.R, weights=temp_df.pixels)
    avg_col[c, 1] = np.average(temp_df.B, weights=temp_df.pixels)
    avg_col[c, 2] = np.average(temp_df.G, weights=temp_df.pixels)

# Calculate weighted average of HSV color

hsv_matrix = np.zeros((len(df), 3))

for i in range(len(df)):
    hsv_matrix[i] = colorsys.rgb_to_hsv(r=df.R[i] / 255, g=df.G[i] / 255, b=df.B[i] / 255)

df['h'] = hsv_matrix[:, 0]
df['s'] = hsv_matrix[:, 1]
df['v'] = hsv_matrix[:, 2]

avg_col2 = np.zeros((kmeans.n_clusters, 3))
for c in range(kmeans.n_clusters):
    temp_df = df[df.kcenter == c]
    avg_col2[c, 0], avg_col2[c, 1], avg_col2[c, 2] = colorsys.hsv_to_rgb(
        h=np.average(temp_df.h, weights=temp_df.pixels),
        s=np.average(temp_df.s, weights=temp_df.pixels),
        v=np.average(temp_df.v, weights=temp_df.pixels))
avg_col2 *= 255




fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(131, projection='3d')

x = kmeans.cluster_centers_[:, 0]
y = kmeans.cluster_centers_[:, 1]
z = kmeans.cluster_centers_[:, 2]
c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
s = 300

ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)
ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)
ax.set_title('RGB K-means', fontsize=16)


ax = fig.add_subplot(132, projection='3d')
x = avg_col[:, 0]
y = avg_col[:, 1]
z = avg_col[:, 2]
c = [hexencode(r) for r in avg_col]
s = 300

ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)
ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)
ax.set_title('Weighted average RGB', fontsize=16)


ax = fig.add_subplot(133, projection='3d')
x = avg_col2[:, 0]
y = avg_col2[:, 1]
z = avg_col2[:, 2]
c = [hexencode(r) for r in avg_col2]
s = 300

ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)
ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)
ax.set_title('Weighted average HSV', fontsize=16)




fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = df.R
y = df.G
z = df.B
c = df.hex
s = 30
ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)


x = kmeans.cluster_centers_[:, 0]
y = kmeans.cluster_centers_[:, 1]
z = kmeans.cluster_centers_[:, 2]
c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
s = 1600
ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1, marker='o')

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)

ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)



#%% HSV k-means

kmeansHSV = KMeans(n_clusters=4, random_state=0, n_init=10).fit(df[['h', 's', 'v']])

dfHSV = df.copy()
dfHSV['kcenter'] = kmeansHSV.labels_


HSVcenters = np.zeros((kmeansHSV.n_clusters, 3))
for i in range(kmeansHSV.n_clusters):
    HSVcenters[i, :] = colorsys.hsv_to_rgb(h=kmeansHSV.cluster_centers_[i, 0],
                                           s=kmeansHSV.cluster_centers_[i, 1],
                                           v=kmeansHSV.cluster_centers_[i, 2])
HSVcenters *= 255



fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = kmeans.cluster_centers_[:, 0]
y = kmeans.cluster_centers_[:, 1]
z = kmeans.cluster_centers_[:, 2]
c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
s = 400

ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1)


x = HSVcenters[:, 0]
y = HSVcenters[:, 1]
z = HSVcenters[:, 2]
c = [hexencode(HSVcenters[i,:]) for i in range(kmeansHSV.n_clusters)]

ax.scatter(x, y, z, c=c, s=s, alpha=1, edgecolor='k', lw=1, marker='s')

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)
ax.set_xlabel('Red', fontsize=14)
ax.set_ylabel('Green', fontsize=14)
ax.set_zlabel('Blue', fontsize=14)
ax.set_title('Comparison of K-means', fontsize=16)
ax.legend(['RGB k-means', 'HSV k-means'], scatterpoints=1, frameon=False, fontsize=13)


#%% HSV plotting

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = df.h
y = df.s
z = df.v
c = df.hex
s = 30

ax.scatter(x, y, z, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)

# ax.set_xlim3d(0, 255)
# ax.set_ylim3d(0, 255)
# ax.set_zlim3d(0, 255)

ax.set_xlabel('H', fontsize=14)
ax.set_ylabel('S', fontsize=14)
ax.set_zlabel('V', fontsize=14)



x = df.h
y = df.s
z = df.v
c = df.hex
s = 30

plt.scatter(x, y, c=c, s=s, alpha=.6, edgecolor='k', lw=0.3)



df[['h', 's', 'v']].describe()


#%% Circular plot of HSV

circ_y = df.s*np.sin(df.h*2*np.pi)
circ_x = df.s*np.cos(df.h*2*np.pi)


plt.figure(figsize=(16,6))
ax = plt.subplot(121)
plt.scatter(circ_x, circ_y, s=40, alpha = .25, c=df.hex, lw=0)
ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    left=False,
    labelleft=False)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

ax = plt.subplot(122)
plt.scatter(df.v*circ_x, df.v*circ_y, s=40, alpha = .25, c=df.hex, lw=0)
ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    left=False,
    labelleft=False)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

#plt.savefig('figures/hsv-proj.png', bbox_inches='tight')



#%%

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = df.v*df.s*np.sin(df.h*2*np.pi)
y = df.v*df.s*np.cos(df.h*2*np.pi)
z = df.v
c = df.hex
s = 30

ax.scatter(x, y, z, c=c, s=s, alpha=.2, edgecolor='k', lw=0)

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

#plt.savefig('figures/hsv-scatter.png', bbox_inches='tight')




#%%

#cut = df[df.v > 0.40]

cut = df


fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = cut.v*cut.s*np.sin(cut.h*2*np.pi)
y = cut.v*cut.s*np.cos(cut.h*2*np.pi)
z = cut.v
c = cut.hex
s = 30

ax.scatter(x, y, z, c=c, s=s, alpha=.2, edgecolor='k', lw=0)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 1)

# ax.set_xlabel('H', fontsize=14)
# ax.set_ylabel('S', fontsize=14)
ax.set_zlabel('V', fontsize=14)


#%% Plotly plotting

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


#%% RGB
# Interactive plot of individual colors and cluster centers for RGB decomposition. Compare cluster centers with weighted average color within each cluster.

weighted_cluster_centers = np.zeros((kmeans.n_clusters, 3))
for c in range(kmeans.n_clusters):
    weighted_cluster_centers[c] = np.average(df[df['kcenter'] == c][['R', 'G', 'B']], weights=df[df['kcenter'] == c]['pixels'], axis=0)




x = df.R
y = df.G
z = df.B
c = df.hex

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=6,
        color=c,
        opacity=0.25),
    name='Individual colors')

x = kmeans.cluster_centers_[:, 0]
y = kmeans.cluster_centers_[:, 1]
z = kmeans.cluster_centers_[:, 2]
c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]

trace2 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=16,
        color=c,
        opacity=1),
    name='k-means center')

x = weighted_cluster_centers[:, 0]
y = weighted_cluster_centers[:, 1]
z = weighted_cluster_centers[:, 2]
c = [hexencode(kmeans.cluster_centers_[i,:]) for i in range(kmeans.n_clusters)]
trace3 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=16,
        color="#aa00aa",
        opacity=1),
    name='weighted center')

data = [trace1, trace2, trace3]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)
_ = iplot(fig)

