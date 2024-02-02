import numpy as np
import pandas as pd
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

