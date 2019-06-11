import pandas as pd

path = '/Users/mac/Downloads/frametest/Results/The.Great.Wall/results/'
df = pd.read_csv(path + 'palette_array.csv')

frames = []
for i in range(max(df['pic'])+1):
    count = len(df.loc[df['pic']==i])

    data = {'pic': i}

    df_part = df.loc[df['pic'] == i]

    per_t = tuple(df_part['percentage'])
    # r_t = tuple(df_part['r'])
    # g_t = tuple(df_part['g'])
    # b_t = tuple(df_part['b'])
    # h_t = tuple(df_part['h'])
    # s_t = tuple(df_part['s'])
    # v_t = tuple(df_part['v'])
    # hh_t = tuple(df_part['hh'])
    # ss_t = tuple(df_part['ss'])
    # ll_t = tuple(df_part['ll'])
    hex_t = tuple(df_part['hex'])

    for j in range(1, 9):
        if j <= count:
            data['percentage%d' % j] = per_t[j - 1]
            # data['r%d' % j] = r_t[j - 1]
            # data['g%d' % j] = g_t[j - 1]
            # data['b%d' % j] = b_t[j - 1]
            # data['h%d' % j] = h_t[j - 1]
            # data['s%d' % j] = s_t[j - 1]
            # data['v%d' % j] = v_t[j - 1]
            # data['hh%d' % j] = hh_t[j - 1]
            # data['ss%d' % j] = ss_t[j - 1]
            # data['ll%d' % j] = ll_t[j - 1]
            data['hex%d' % j] = hex_t[j - 1]

    frames.append(pd.DataFrame(data, index=[0]))

df0 = pd.concat(frames, ignore_index=True, sort=False)

df0.to_csv(path + 'palette_array_widthwise_8_hex.csv', index=False)
