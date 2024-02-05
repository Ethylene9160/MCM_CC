from ethy.model.MCModel import MCModel
from ethy.model.MCMSVM import MSMSVM as MCMSVM
import pandas as pd
from sklearn.svm import SVR  # 支持向量回归
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import python.data_reader as mDR

def read_and_combine_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def append_momentum(combined_df, max_shift):
    player_list = []
    for index, row in combined_df.iterrows():
        player_data = row.to_dict()
        player_list.append(player_data)
    p1m, p2m = mDR.getMomentum(player_list)
    # 将momentum趋势添加到DataFrame中
    combined_df['Player1_Momentum'] = p1m
    combined_df['Player2_Momentum'] = p2m
    for i in range(1, max_shift):
        combined_df[f'Player1_Momentum_shift_{i}'] = combined_df['Player1_Momentum'].shift(i).fillna(0)
        combined_df[f'Player2_Momentum_shift_{i}'] = combined_df['Player2_Momentum'].shift(i).fillna(0)
    combined_df['Player1_Momentum_shift_pre'] = combined_df['Player1_Momentum'].shift(-1).fillna(0)
    combined_df['Player2_Momentum_shift_pre'] = combined_df['Player2_Momentum'].shift(-1).fillna(0)
    return combined_df

def SVM_data_process(file_path, max_shift):
    combined_df = read_and_combine_data(file_path)
    # 在最后一列加上计算好的两个玩家的momentum
    combined_df = append_momentum(combined_df, max_shift)
    # 准备训练数据 -> 当前的特征+之前五个的momentum
    features = combined_df[[
                            # 'p1_points_won', 'p2_points_won', \
                            # 'p1_double_fault', 'p2_double_fault', \
                            # 'p1_distance_run', 'p2_distance_run', \
                            # 'p1_break_pt_won', 'p2_break_pt_won', \
                            # 'p1_break_pt_missed', 'p2_break_pt_missed', \
                            # 'p1_unf_err','p2_unf_err',\
                            # 'p1_sets', 'p2_sets', \
                            # 'p1_games', 'p2_games', \
                            # 'p1_score', 'p2_score', \
                            #
                            # 'p1_winner', 'p1_ace', \
                            # 'Player1_Momentum', \
                            'Player1_Momentum_shift_1', \
                            'Player1_Momentum_shift_2', \
                            'Player1_Momentum_shift_3', \
                            'Player1_Momentum_shift_4']]
    labels = combined_df['Player1_Momentum']
    return features, labels

# def perform_sensitivity_analysis(X_train, y_train, X_test, y_test, param_grid):
#     results = []
#     for params in ParameterGrid(param_grid):
#         svm_model = SVR(**params)
#         svm_model.fit(X_train, y_train)
#         predictions = svm_model.predict(X_test)
#         r2 = r2_score(y_test, predictions)
#         results.append((params, r2))
#
#     # Debug information
#     print("Sensitivity Analysis Results:")
#     for result in results:
#         print(f'Parameters: {result[0]}, R^2 Score: {result[1]}')
#
#     # Visualize 散点图
#     param_names = list(param_grid.keys())
#     for i in range(len(param_names)):
#         param_name = param_names[i]
#         values = [result[0][param_name] for result in results]
#         r2_scores = [result[1] for result in results]
#
#         plt.figure(figsize=(8, 5))
#         plt.plot(values, r2_scores, marker='o')
#         plt.title(f'Sensitivity Analysis for {param_name}')
#         plt.xlabel(param_name)
#         plt.ylabel('R^2 Score')
#         plt.show()


def perform_sensitivity_analysis(X_train, y_train, X_test, y_test, param_grid):
    results = []
    for params in ParameterGrid(param_grid):
        svm_model = MCMSVM()
        # svm_model0 = SVR(**params)
        # svm_model.svm_model = svm_model0
        # # svm_model.fit(X_train, y_train)
        # svm_model.train(X_train, y_train)
        # svm_model.save('svm_model_'+params['kernel']+str(params['C'])+str(params['epsilon'])+'.ju')
        #训练结束后，直接读取的方法：
        svm_model = svm_model.load('svm_model_'+params['kernel']+str(params['C'])+str(params['epsilon'])+'.ju')
        predictions = svm_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        results.append((params, r2))

    # Debug information
    print("Sensitivity Analysis Results:")
    for result in results:
        print(f'Parameters: {result[0]}, R^2 Score: {result[1]}')

    # Visualize Heatmap

    param_names = list(param_grid.keys())
    for i in range(len(param_names)):
        param_name = param_names[i]
        values = [result[0][param_name] for result in results]
        r2_scores = [result[1] for result in results]

        plt.figure(figsize=(8, 5))

        plt.scatter(values, r2_scores, c=r2_scores, cmap='viridis', marker='o')  # Use scatter for heatmap effect
        plt.title(f'SVR Sensitivity Analysis: {param_name}', fontdict={'family':'Times New Roman', 'size':14})
        plt.xlabel(param_name, fontproperties='Times New Roman', size=14)
        plt.ylabel('R^2 Score', fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.xticks(fontproperties='Times New Roman', size=14)

        plt.tick_params(axis='y', labelsize=14)
        # plt.colorbar(label='R^2 Score', fontdict={'family':'Times New Roman','size': 12})  # Add colorbar
        cb = plt.colorbar()
        cb.set_label('R^2 Score', fontdict={'family': 'Times New Roman', 'size': 14})

        plt.savefig(f'SVR_Sensitivity_Analysis_for_{param_name}.eps')

        plt.show()
    # 柱状图


class MSMSVM(MCModel):
    def __init__(self, kernel='rbf', C=3, epsilon=0.001):
        # 训练SVM模型
        super().__init__()
        self.svm_model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def train(self, X, y):
        self.svm_model.fit(X, y)

    def predict(self, X):
        return self.svm_model.predict(X)

    def getModelParam(self):
        pass

def SVM_main():
    # 使用方法：
    model = MSMSVM(kernel='rbf', C=3, epsilon=0.001)

    # 读取数据
    X_train, y_train = SVM_data_process(['../../statics/training/session_train.csv'], 5)
    X_test, y_test = SVM_data_process(['../../statics/training/session_test.csv'], 5)

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'epsilon': [0.01, 0.1, 0.2, 0.3 ,0.35, 0.4, 0.45, 0.5, 0.7, 0.9]
    }
    print("aa")
    perform_sensitivity_analysis(X_train, y_train, X_test, y_test, param_grid)

    # model.train(X_train, y_train)
    # # model = model.load('../model_params/lr_model.pkl')
    # predictions = model.predict(X_test)
    # r2 = r2_score(y_test, predictions)
    # print(f'R^2 Score: {r2}')
    # #draw pic.
    # plt.figure(figsize=(8, 5))
    # plt.plot(y_test, label='True')
    # plt.plot(predictions, label='Predict')
    # plt.legend()
    # plt.show()
    # print('Done')
    #
    # print('loss:', model.MSELoss(y_test, predictions))
    # accuracy, precision, recall, f1 = model.judge(y_test, predictions)
    # print('accuracy:', accuracy)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1:', f1)
    #
    # # model.save('../model_params/SVM_Model.pkl')  # 传入文件路径进行保存

if __name__ == '__main__':
    SVM_main()
