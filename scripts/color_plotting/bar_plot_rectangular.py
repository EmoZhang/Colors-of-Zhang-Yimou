import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = '/Users/mac/Downloads/frametest/csv/Red.Sorghum/'
df = pd.read_csv(path + 'palette_array_cr.csv')

percentage = np.array(df.loc[:, 'percentage1'].fillna(1.))
c = df.hex1.fillna('#000000')

# percentage = np.array(df.loc[:, 'percentage1'])
# c = df.hex1

fig, ax = plt.subplots()

size = 0.2
vals = np.arange(1, len(df)+1)
# normalize vals to 2 pi
valsnorm = vals/np.sum(vals)*8
# obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=0, height=size*percentage,
       color=c, edgecolor='w', linewidth=0, align="edge")

max_per = max(percentage)
max_height = size * max_per + 1 - 2.75*size

plt.ylim((0, 0.3))
plt.xticks([0], ('00:00',))
plt.yticks([0.475, ], ('100%',))

# ax.set(title="")
# ax.set_axis_off()
plt.show()
# plt.savefig(path + 'dominant_bar_plot_1st.png')
