import sys
from datetime import datetime
from HSVK1 import *


def main():
    """脚本主函数"""

    print('If you want exit, just type in "exit".')

    # 输入截图文件夹的路径
    while True:
        # path = input('Please type in the path of your picture folder: ')
        path = '/Users/mac/Downloads/frametest/test'

        # 输入"exit"退出程序
        if path == 'exit':
            sys.exit()

        # 判断输入路径是否正确/是否存在/是否为文件夹
        if not os.path.exists(path):
            print('The path does not exist! Please check again.')
        else:
            if not os.path.isdir(path):
                print('The path does not refer to a folder! Please check again.')
            else:
                if path[-1] != '/':
                    path = path + '/'
                    break
                else:
                    break

    print('Loading pictures...')
    files = load_pics(path)

    print('Getting K-means data...')
    df_all = get_data(path, files)

    print('Getting final K-means data...')
    kmeans_final(df_all, path, method='GS')

    print('Mission Complete!')


if __name__ == '__main__':
    t_s = datetime.now()
    main()
    t_e = datetime.now()
    usedtime = t_e - t_s
    print('[%s]' % usedtime)
