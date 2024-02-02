import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import python.PCA as mPCA
import python.SVM as mSVM

map = {}

def load_data():
    data = pd.read_csv('statics/data_dictionary.csv')
    return data

def read_data(data_list):
    data = pd.read_csv('statics/data.csv')

    return data