import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import python.PCA as mPCA
import python.data_reader as mDR
from python.data_reader import hint

data_path = 'statics/29splits/session1.csv'
match_map = {}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    x = sigmoid(x)
    return x * (1 - x)  # sigmoid函数的导数

if __name__ == '__main__':
    # 加载CSV数据
    data = mDR.read_data(data_path)
    #
    # # 直接将DataFrame中的每行转换为字典并添加到playerList
    player_list = []
    for index, row in data.iterrows():
        player_data = row.to_dict()
        player_list.append(player_data)

    # print('match map:', match_map)
    # # 打印playerList中的前几个条目以进行验证
    for player in player_list[:2]:
        print(player)
    plist = mDR.getList(data_path)
    for player in plist[:2]:
        print(player)
    mPCA_test = mPCA.PCA()
    # 取出plist的第5列及以后的元素
    plist = np.array(plist)
    # print(plist.shape)
    plist = plist[:, 5:]

    mPCA_test.train(plist[1:2000,:])
    # out = mPCA_test.transform(plist, 8)
    mPCA_test.draw_variance_plot()
    mPCA_test.draw_split_variance_plot()
    # new_list = mPCA_test.transform(plist, 8)
    principle = mPCA_test.transform(plist, 1)

    # 映射为0-1分布
    principle = (principle - np.min(principle)) / (np.max(principle) - np.min(principle))
    # print(principle)
    # plt.figure()
    # plt.plot(principle)
    # plt.show()

    p1m, p2m = mDR.getMomentum(player_list)
    # 在matlab中，绘制出momentum的两条曲线的变化啊趋势。
    plt.figure()
    plt.plot(p1m, label='p1')
    plt.plot(p2m, label='p2')
    # 标注哪根曲线是第一列，哪一根是第二列
    plt.legend()
    plt.show()

    # size = 0
    # cor = 0
    # for i in range(len(reconstruct_list)):
    #     for j in range(len(reconstruct_list[i])):
    #         if reconstruct_list[i][j] != plist[i][j]:
    #             pass
    #             # print('reconstruct_list[', i, '][', j, ']:', reconstruct_list[i][j])
    #             # print('plist[', i, '][', j, ']:', plist[i][j])
    #         else:
    #             cor += 1
    #         size += 1
    #
    # print('correct rate:', cor / size)



