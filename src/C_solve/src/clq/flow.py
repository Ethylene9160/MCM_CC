import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import python.data_reader as mDR
from scipy.io import savemat
import corr_Method as cm
from scipy.io import savemat

data = mDR.read_data("../statics/29splits/session1.csv")
player_list = []

for index, row in data.iterrows():
    player_data = row.to_dict()
    player_list.append(player_data)



err1= cm.get_error_sum(player_list,1)
err2= cm.get_error_sum(player_list,2)
serve1 = cm.get_server_change(player_list,1)
serve2 = cm.get_server_change(player_list,2)
surprise1 = cm.get_surprise_sum(player_list,1)
surprise2 = cm.get_surprise_sum(player_list,2)

savemat('output/corr.mat', {'err1': err1, 'err2': err2, 'surprise1': surprise1, 'surprise2': surprise2, 'server1': serve1,'server2': serve2})