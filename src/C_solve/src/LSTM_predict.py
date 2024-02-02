import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import LSTM_Method
import pandas as pd

data_path = 'statics/hot_data.csv'
data = {}
historical = 5

try:
    original_data = pd.read_csv(data_path)
    for index, row in original_data.iterrows():
        match_id = row['Q1-match_id']
        if match_id not in data:
            data[match_id] = []
        data[match_id].append([row['Q17-p1_points_won'], row['Q18-p2_points_won']])
except Exception as e:
    print(e)

features, labels, seq_num = LSTM_Method.slide_windows(data['1'], historical)
for i in range(1,len(data)):
    next_features, next_labels,next_seq_num = LSTM_Method.slide_windows(data['i'], historical)
    features = torch.cat((features, next_features), dim=0)  # 连接起来
    labels = torch.cat((labels, next_labels), dim=0)

