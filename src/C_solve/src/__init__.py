import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import python.PCA as mPCA
import python.SVM as mSVM
import python.data_reader as mDR
from python.data_reader import hint

data_path = 'statics/hot_data.csv'
match_map = {}

def read_data(csv_file_path):
    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file_path)
    return data

def read_single_data(single_line):
    # 将单行数据转换为字典
    single_map = {}
    for i, value in enumerate(single_line):
        single_map[hint[i]] = value
    #如果没有这个选手的数据，就添加进去
    if single_line[0] not in match_map:
        match_map[single_line[0]] = 1
    return single_map


if __name__ == '__main__':
    # 加载CSV数据
    # data = read_data(data_path)
    #
    # # 直接将DataFrame中的每行转换为字典并添加到playerList
    # player_list = []
    # for index, row in data.iterrows():
    #     player_data = row.to_dict()
    #     player_list.append(player_data)
    # print('match map:', match_map)
    # # 打印playerList中的前几个条目以进行验证
    # for player in player_list[:2]:
    #     print(player)
    plist = mDR.getList(data_path)
    for player in plist[:2]:
        print(player)
    mPCA_test = mPCA.PCA()
    mPCA_test.train(plist[1:100])
    # out = mPCA_test.transform(plist, 8)
    mPCA_test.draw_variance_plot()
    mPCA_test.draw_split_variance_plot()
    new_list = mPCA_test.transform(plist, 8)
    reconstruct_list = mPCA_test.inverse_transform(new_list, 8)

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



