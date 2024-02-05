# 示例数据
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import python.data_reader as mDR
from scipy.io import savemat
import corr_Method as cm

sucess_data = mDR.read_data("../statics/corr.csv")
whole_data = mDR.read_data("../statics/29splits/session1.csv")

"""=====================定义变量====================="""
player_list = []  # 包含所有信息的列表，通过get_data函数获取某一列的信息
p1s = []  # p1连续得分情况
bi_p1s = []  # p1连续得分情况，0表示未连续得分，1表示连续得分
# z       # z分数

"""=====================验证随机性====================="""

for index, row in whole_data.iterrows():
    player_data = row.to_dict()
    player_list.append(player_data)

'''获取p1连续得分情况'''
# 1表示p1连续得分，0表示p1未连续得分
victor = cm.get_data(player_list, 'point_victor')
bi_p1s = [x if x == 1 else 0 for x in victor]
j = 0
for i in range(len(victor)):
    if victor[i] == 1:
        j += 1
    else:
        j = 0
    p1s.append(j)

# z = cm.z_score(bi_p1s)
#
# # print(victor[:10])
# # print(p1s[:10])
# # print(z)

"""=====================验证相关性====================="""
err= cm.error_rate(player_list)
serve = cm.get_server(player_list)
surprise = cm.get_surprise(player_list)


# a = []

# for i in range(len(serve)):
#     a.append(-0.5*err[i] + 0.5*serve[i] + 0.5*surprise[i])
# p1m = mDR.getMomentum(player_list)[0]
#
# corr1, p_value1 = pearsonr(bi_p1s, err)
# corr2, p_value2 = pearsonr(bi_p1s,serve)
# corr3, p_value3 = pearsonr(bi_p1s,surprise)
# corr4, p_value4 = pearsonr(bi_p1s,a)
# corr5, p_value5 = pearsonr(bi_p1s,p1m)

# 打印结果
# print("Pearson Correlation Coefficient:", corr1)
# print("P-value:", p_value1)
# print("Pearson Correlation Coefficient:", corr2)
# print("P-value:", p_value2)
# print("Pearson Correlation Coefficient:", corr3)
# print("P-value:", p_value3)
# print("Pearson Correlation Coefficient:", corr4)
# print("P-value:", p_value4)


