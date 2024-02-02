import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import python.PCA as mPCA
import python.SVM as mSVM
from python.data_reader import hint

data_path = 'statics/Wimbledon_featured_matches.csv'


def read_data(csv_file_path):
    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file_path)
    return data

def read_single_data(single_line):
    # 将单行数据转换为字典
    single_map = {}
    for i, value in enumerate(single_line):
        single_map[hint[i]] = value
    return single_map


if __name__ == '__main__':
    # 加载CSV数据
    data = read_data(data_path)

    # 直接将DataFrame中的每行转换为字典并添加到playerList
    player_list = []
    for index, row in data.iterrows():
        player_data = row.to_dict()
        player_list.append(player_data)

    # 打印playerList中的前几个条目以进行验证
    for player in player_list[:2]:
        print(player)

