import pandas as pd
import numpy as np


def relative_luminance(sRGB):

    RsRGB, GsRGB, BsRGB = sRGB / 255

    if RsRGB <= 0.03928:
        R = RsRGB / 12.92
    else:
        R = ((RsRGB+0.055) / 1.055) ** 2.4

    if GsRGB <= 0.03928:
        G = GsRGB / 12.92
    else:
        G = ((GsRGB+0.055) / 1.055) ** 2.4

    if BsRGB <= 0.03928:
        B = BsRGB / 12.92
    else:
        B = ((BsRGB+0.055) / 1.055) ** 2.4

    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    return L


def contrast_ratio(L1, L2):
    Lmax = max(L1, L2)
    Lmin = min(L1, L2)
    cr = (Lmax + 0.05) / (Lmin + 0.05)
    if L1 < L2:
        cr = -cr

    return cr


path = '/Users/mac/Downloads/frametest/csv/The.Great.Wall/'
df = pd.read_csv(path + 'palette_array_widthwise.csv')

color1 = np.array(df.loc[:, 'r1': 'b1'])
color2 = np.array(df.loc[:, 'r2': 'b2'])

cr_list = []
for i in range(max(df['pic'])+1):
    rl1 = relative_luminance(color1[i])
    rl2 = relative_luminance(color2[i])
    cr0 = contrast_ratio(rl1, rl2)
    cr_list.append(cr0)

col_name = df.columns.tolist()
col_name.insert(1, 'contrast_ratio')
df = df.reindex(columns=col_name)

df['contrast_ratio'] = cr_list

df.to_csv(path + 'palette_array_cr.csv', index=False)
