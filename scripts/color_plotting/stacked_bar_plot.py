import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = '/Users/mac/Downloads/frametest/csv/Coming.Home/'
df = pd.read_csv(path + 'palette_array_widthwise_8_hex.csv')

percentage = np.array(df.loc[:, 'percentage1'].fillna(1.))
c = df.hex1.fillna('#000000')

fig, ax = plt.subplots(subplot_kw=dict(polar=True))
fig.set_size_inches(10, 10, forward=True)

size = 0.3
vals = np.arange(1, len(df) + 1)
# normalize vals to 2 pi
valsnorm = vals / np.sum(vals) * 2 * np.pi
# obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)
ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=1 - 2.75 * size, height=size * percentage,
       color=c, edgecolor='w', linewidth=0, align="edge")

bottom = 1 - 2.75 * size + percentage * size

for i in range(2, 9):
    percentage = np.array(df.loc[:, 'percentage%d' % i].fillna(0))
    c = df.loc[:, 'hex%d' % i].fillna('#000000')

    vals = np.arange(1, len(df) + 1)
    # normalize vals to 2 pi
    valsnorm = vals / np.sum(vals) * 2 * np.pi
    # obtain the ordinates of the bar edges
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

    ax.bar(x=valsleft.flatten(),
           width=valsnorm.flatten(), bottom=bottom, height=size * percentage,
           color=c, edgecolor='w', linewidth=0, align="edge")

    bottom = bottom + percentage * size

# plt.ylim((0, 0.475))
plt.xticks([0], ('00:00',))
# plt.yticks([0.475, ], ('100%',))
plt.yticks([])

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# ax.set(title="")
# ax.set_axis_off()
# plt.show()
plt.savefig(path + 'palette_bar_plot.png')
