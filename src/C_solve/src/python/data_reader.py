import numpy as np
import pandas as pd
pd.set_option('display.notebook_repr_html',False)
hint = [
    'match_id',
    'player1',
    'player2',
    'elapsed_time',
    'set_no',
    'game_no',
    'point_no',
    'p1_sets',
    'p2_sets',
    'p1_games',
    'p2_games',
    'p1_score',
    'p2_score',
    'server',
    'serve_no',
    'point_victor',
    'p1_points_won',
    'p2_points_won',
    'game_victor',
    'set_victor',
    'p1_ace',
    'p2_ace',
    'p1_winner',
    'p2_winner',
    'winner_shot_type',
    'p1_double_fault',
    'p2_double_fault',
    'p1_unf_err',
    'p2_unf_err',
    'p1_net_pt',
    'p2_net_pt',
    'p1_net_pt_won',
    'p2_net_pt_won',
    'p1_break_pt',
    'p2_break_pt',
    'p1_break_pt_won',
    'p2_break_pt_won',
    'p1_break_pt_missed',
    'p2_break_pt_missed',
    'p1_distance_run',
    'p2_distance_run',
    'rally_count',
    'speed_mph',
    'serve_width',
    'serve_depth',
    'return_depth'
]


def read_data(csv_file_path):
    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file_path)
    return data
def read_excel_data(excel_file_path):
    # 从excel文件中读取数据
    data = pd.read_excel(excel_file_path)
    return data
def read_single_data(single_line):
    # 将单行数据转换为字典
    single_map = {}
    for i, value in enumerate(single_line):
        single_map[hint[i]] = value
    return single_map

def line_to_data(single_line):
    # 将单行数据转换为列表
    single_list = []
    for i, value in enumerate(single_line):
        if i == 0:
            continue
        single_list.append(value)
    return single_list

def getList(data_path):
    # 加载CSV数据
    data = read_data(data_path)

    # 直接将DataFrame中的每行转换为字典并添加到playerList
    data_list = []
    for index, row in data.iterrows():
        data_list.append(line_to_data(row))
    return data_list

def getExcelList(data_path):
    # 加载excel数据
    data = read_excel_data(data_path)

    # 直接将DataFrame中的每行转换为字典并添加到playerList
    data_list = []
    for index, row in data.iterrows():
        data_list.append(line_to_data(row))
    return data_list

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    x = sigmoid(x)
    return x * (1 - x)  # sigmoid函数的导数
# 计算当前势头。
#
def getMots(single_map, lastP1, lastP2, p1_continue = 0, p2_continue = 0, p1_count = 0, p2_count = 0):
    l11 = int(single_map['p1_points_won']) - int(single_map['p2_points_won'])
    l12 = single_map['p2_points_won'] - single_map['p1_points_won']
    w1 = 0.05
    # w1 = 0
    l21 = 0.3*single_map['p1_winner'] + 0.2*single_map['p1_ace']
    l22 = 0.3*single_map['p2_winner'] + 0.2*single_map['p2_ace']
    w2 = 0.6
    # w2 = 0
    l31 = single_map['p1_double_fault'] - single_map['p2_double_fault']
    l32 = single_map['p2_double_fault'] - single_map['p1_double_fault']
    w3 = -0.25
    # w3 = 0
    l41 = single_map['p1_distance_run'] - single_map['p2_distance_run']
    l42 = single_map['p2_distance_run'] - single_map['p1_distance_run']
    w4 = -0.004
    # w4 = 0
    l51 = single_map['p1_break_pt_won'] - single_map['p2_break_pt_won']
    l52 = single_map['p2_break_pt_won'] - single_map['p1_break_pt_won']
    w5 = 0.5
    w5 = 0
    l61 = single_map['p1_break_pt_missed'] - single_map['p2_break_pt_missed']
    l62 = single_map['p2_break_pt_missed'] - single_map['p1_break_pt_missed']
    w6 = -0.5
    # w6 = 0
    l71 = single_map['p1_unf_err'] - single_map['p2_unf_err']
    l72 = single_map['p2_unf_err'] - single_map['p1_unf_err']
    w7 = 0.5
    # w7 = 0

    l81 = single_map['p1_sets']**2.5 - single_map['p2_sets']*2.5
    l82 = -l81
    w8 = 0.10
    # w8 = 0
    l91 = single_map['p1_games']**2-single_map['p2_games']**2
    l92 = -l91
    w9 = 0.3
    # w9 = 0
    l101 = single_map['p1_score']-single_map['p2_score']
    l102 = single_map['p2_score']-single_map['p1_score']
    w10 = 0.2
    # w10=0
    wc = 1.5
    wcontinue = 1.5
    fh = 0.25
    bc = 1-fh
    p1 = w1*l11+w2*l21+w3*l31+w4*l41+w5*l51+w6*l61+w7*l71+w8*l81+w9*l91+w10*l101+wc*p1_count+wcontinue*p1_continue
    p2 = w1*l12+w2*l22+w3*l32+w4*l42+w5*l52+w6*l62+w7*l72+w8*l82+w9*l92+w10*l102+wc*p2_count+wcontinue*p2_continue
    p1 = fh * p1 + bc * lastP1
    p2 = fh * p2 + bc * lastP2
    mtp = (np.exp(p1) + np.exp(p2))
    return np.exp(p1)/mtp, np.exp(p2)/mtp

def getMomentum(whole_list):
    p1_monmentum_list = []
    p2_monmentum_list = []
    p1, p2 = getMots(whole_list[0],0,0)
    p1_count = 0
    p2_count = 0
    p1_continue = 0
    p2_continue = 0
    p1_monmentum_list.append(p1)
    p2_monmentum_list.append(p2)

    time_count = 5
    for i in range(1,len(whole_list)):
        if whole_list[i]['point_victor'] == 1:
            # p1_count = tanh(p1_count+d_tanh(p1_count))
            p1_count += 1
            p1_continue += 1
            p2_continue -= 1
            if p1_continue > time_count:
                p1_continue = time_count
                p2_continue = 0
            p2_count = 0
        else:
            # p2_count = tanh(p1_count+d_tanh(p2_count))
            p2_count += 1
            p2_continue += 1
            p1_continue -= 1
            if p2_continue > time_count:
                p2_continue = time_count
                p1_continue = 0
            p1_count = 0
        p1, p2 = getMots(whole_list[i],p1,p2,sigmoid(p1_continue-3),sigmoid(p2_continue-3),sigmoid(p1_count-5),sigmoid(p2_count-5))
        p1_monmentum_list.append(p1)
        p2_monmentum_list.append(p2)
    return p1_monmentum_list, p2_monmentum_list
def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def read_data(csv_file_path):
    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file_path)
    return data

def read_single_data(single_line):
    # 将单行数据转换为字典
    single_map = {}
    for i, value in enumerate(io=single_line):
        single_map[hint[i]] = value
    #如果没有这个选手的数据，就添加进去
    # if single_line[0] not in match_map:
    #     match_map[single_line[0]] = 1
    return single_map

def getP1P2SetScore(wholeMap):
    # p1_score = []
    # p2_score = []
    # for single_map in wholeMap:
    #     p1_score.append(single_map['p1_sets'])
    #     p2_score.append(single_map['p2_sets'])
    # return p1_score, p2_score
    return getP1P2Sets(wholeMap, 'p1_sets', 'p2_sets')

def getP1P2Sets(wholeMap, p1_index, p2_index):
    p1_list = []
    p2_list = []
    for single_map in wholeMap:
        p1_list.append(single_map[p1_index])
        p2_list.append(single_map[p2_index])
    return p1_list, p2_list

