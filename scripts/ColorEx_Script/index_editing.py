import pandas as pd

# 输入csv所在的路径和文件名，并读取csv
path = '/Users/mac/Downloads/frametest/csv/Red.Sorghum/'
df = pd.read_csv(path + 'palette_array_widthwise.csv')

# 截图去掉开头后，从第几个编号开始，如从image-00110.jpg开始，则begin = 110
begin = 110

usage = []
for i in range(len(df)):
    usage.append(None)


def add_usage(n, start, stop):
    start = start - begin
    stop = stop - begin
    for i in range(stop - start + 1):
        usage[start + i] = n


# 给表格中的指定段落标号，共有三个参数：段落编号，第一张图片编号，最后一张图片编号
# 例如，第1段，第一张图片为image-00251.jpg，最后一张图片为image-01037.jpg，则如下所示
# add_usage(1, 251, 1037)
add_usage(1, 251, 1037)
add_usage(2, 1135, 1174)
add_usage(3, 2565, 2565)
add_usage(4, 2591, 2591)
add_usage(5, 3624, 4719)
add_usage(6, 9362, 9362)
add_usage(7, 10454, 10605)
add_usage(8, 10701, 10768)

usage_df = pd.DataFrame(data={'usage': usage})
df0 = pd.concat([usage_df, df], axis=1, sort=False)

# 输出csv
df0.to_csv(path + 'palette_array_usage.csv', index=False)
