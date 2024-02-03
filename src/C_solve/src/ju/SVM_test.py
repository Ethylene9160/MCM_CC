import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR  # 支持向量回归
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import python.data_reader as mDR
data_path = 'statics/29splits/session2.csv'

file_paths = [
    'statics/29splits/session1.csv',
    'statics/29splits/session2.csv',
    'statics/29splits/session3.csv',
    'statics/29splits/session4.csv'
]
# 假设calculate_momentum是一个可用的函数，可以计算momentum
# 从您的描述来看，这个函数可能需要根据实际情况进行调整或重写

def read_and_combine_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def append_momentum(combined_df):
    player_list = []
    for index, row in combined_df.iterrows():
        player_data = row.to_dict()
        player_list.append(player_data)
    p1m, p2m = mDR.getMomentum(player_list)
    # 将momentum趋势添加到DataFrame中
    combined_df['Player1_Momentum'] = p1m
    combined_df['Player2_Momentum'] = p2m
    for i in range(1, 5):
        combined_df[f'Player1_Momentum_shift_{i}'] = combined_df['Player1_Momentum'].shift(i).fillna(0)
        combined_df[f'Player2_Momentum_shift_{i}'] = combined_df['Player2_Momentum'].shift(i).fillna(0)
    combined_df['Player1_Momentum_shift_pre'] = combined_df['Player1_Momentum'].shift(-1).fillna(0)
    combined_df['Player2_Momentum_shift_pre'] = combined_df['Player2_Momentum'].shift(-1).fillna(0)
    return combined_df

def train_predict_plot(X_train1, y_train1, X_val1, y_val1, X_train2, y_train2, X_val2, y_val2):
    # 数据标准化
    scaler = StandardScaler()
    features_scaled1 = scaler.fit_transform(X_train1)
    test_features_scaled1 = scaler.transform(X_val1)

    print("===============Training SVM for Player1================")
    # 训练SVM模型
    svm_model1 = SVR()
    svm_model1.fit(features_scaled1, y_train1)

    print("===============Predicting SVM for Player1================")
    svm_predictions1 = svm_model1.predict(test_features_scaled1)
    svm_r2_1 = r2_score(y_val1, svm_predictions1)
    print(f'SVM R^2 Score for Player1: {svm_r2_1}')

    # 数据标准化
    features_scaled2 = scaler.fit_transform(X_train2)
    test_features_scaled2 = scaler.transform(X_val2)

    print("===============Training SVM for Player2================")
    # 训练SVM模型
    svm_model2 = SVR()
    svm_model2.fit(features_scaled2, y_train2)

    print("===============Predicting SVM for Player2================")
    svm_predictions2 = svm_model2.predict(test_features_scaled2)
    svm_r2_2 = r2_score(y_val2, svm_predictions2)
    print(f'SVM R^2 Score for Player2: {svm_r2_2}')

    # 绘制真实值和预测值的图表
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(y_val1.index, y_val1, label='True Values Player1', marker='o')
    plt.plot(y_val1.index, svm_predictions1, label='Predicted Values Player1', marker='x')
    plt.title('True vs Predicted Player1_Momentum')
    plt.xlabel('Index')
    plt.ylabel('Player1_Momentum')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y_val2.index, y_val2, label='True Values Player2', marker='o')
    plt.plot(y_val2.index, svm_predictions2, label='Predicted Values Player2', marker='x')
    plt.title('True vs Predicted Player2_Momentum')
    plt.xlabel('Index')
    plt.ylabel('Player2_Momentum')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    combined_df = read_and_combine_data(file_paths)
    #在最后一列加上计算好的两个玩家的momentum
    combined_df = append_momentum(combined_df)
    # 准备训练数据 -> 当前的特征+之前五个的momentum
    features = combined_df[['p1_unf_err', 'p1_break_pt_missed', 'p1_winner', 'p1_break_pt_won', \
                            'Player1_Momentum','Player1_Momentum_shift_1', 'Player1_Momentum_shift_2',\
                            'Player1_Momentum_shift_3', 'Player1_Momentum_shift_4']]
    labels = combined_df['Player1_Momentum_shift_pre']

    # 读取测试集
    test_df = pd.read_csv('statics/29splits/session5.csv')
    # 对测试集同样做添加数据处理
    test_df = append_momentum(test_df)

    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    test_features_scaled = scaler.transform(test_df[features.columns])

    # 划分训练集和验证集
    # X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    X_train1 = combined_df[['p1_unf_err', 'p1_break_pt_missed', 'p1_winner', 'p1_break_pt_won',
                           'Player1_Momentum', 'Player1_Momentum_shift_1', 'Player1_Momentum_shift_2',
                           'Player1_Momentum_shift_3', 'Player1_Momentum_shift_4']]
    y_train1 = combined_df['Player1_Momentum_shift_pre']
    # 测试集
    X_val1 = test_df[['p1_unf_err', 'p1_break_pt_missed', 'p1_winner', 'p1_break_pt_won',
                      'Player1_Momentum', 'Player1_Momentum_shift_1', 'Player1_Momentum_shift_2',
                      'Player1_Momentum_shift_3', 'Player1_Momentum_shift_4']]
    y_val1 = test_df['Player1_Momentum_shift_pre']

    X_train2 = combined_df[['p2_unf_err', 'p2_break_pt_missed', 'p2_winner', 'p2_break_pt_won',
                           'Player2_Momentum','Player2_Momentum_shift_1', 'Player2_Momentum_shift_2',
                           'Player2_Momentum_shift_3', 'Player2_Momentum_shift_4']]
    y_train2 = combined_df['Player2_Momentum_shift_pre']
    # 测试集
    X_val2 = test_df[['p2_unf_err', 'p2_break_pt_missed', 'p2_winner', 'p2_break_pt_won',
                      'Player2_Momentum', 'Player2_Momentum_shift_1', 'Player2_Momentum_shift_2',
                      'Player2_Momentum_shift_3', 'Player2_Momentum_shift_4']]
    y_val2 = test_df['Player2_Momentum_shift_pre']

    print("===============Training SVM for Player1================")
    # 训练SVM模型
    svm_model = SVR()
    svm_model.fit(X_train1, y_train1)
    print("===============Predicting SVM for Player1================")
    svm_predictions = svm_model.predict(X_val1)
    # svm_accuracy = accuracy_score(y_val, svm_predictions)
    # print(f'SVM Accuracy: {svm_accuracy}')
    svm_r2 = r2_score(y_val1, svm_predictions)
    print(f'SVM R^2 Score: {svm_r2}')

    print("===============Training SVM for Player2================")
    # 训练SVM模型
    svm_model = SVR()
    svm_model.fit(X_train2, y_train2)
    print("===============Predicting SVM for Player2================")
    svm_predictions = svm_model.predict(X_val2)
    # svm_accuracy = accuracy_score(y_val, svm_predictions)
    # print(f'SVM Accuracy: {svm_accuracy}')
    svm_r2 = r2_score(y_val2, svm_predictions)
    print(f'SVM R^2 Score: {svm_r2}')

    train_predict_plot(X_train1, y_train1, X_val1, y_val1, X_train2, y_train2, X_val2, y_val2)

    # # 训练Logistic Regression模型
    # lr_model = LogisticRegression()
    # lr_model.fit(X_train, y_train)
    # lr_predictions = lr_model.predict(X_val)
    # lr_accuracy = accuracy_score(y_val, lr_predictions)
    # print(f'Logistic Regression Accuracy: {lr_accuracy}')

if __name__ == '__main__':
    main()