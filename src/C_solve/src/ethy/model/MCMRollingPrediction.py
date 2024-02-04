from random import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import pandas as pd
import python.data_reader as mDR
from ethy.model.MCModel import MCModel


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
    # 将momentum列添加到DataFrame中
    combined_df['Player1_Momentum'] = p1m
    combined_df['Player2_Momentum'] = p2m
    for i in range(1, max_shift+1):
        combined_df[f'Player1_Momentum_shift_{i}'] = combined_df['Player1_Momentum'].shift(i).fillna(0)
        combined_df[f'Player2_Momentum_shift_{i}'] = combined_df['Player2_Momentum'].shift(i).fillna(0)
    combined_df['Player1_Momentum_shift_pre'] = combined_df['Player1_Momentum'].shift(-1).fillna(0)
    combined_df['Player2_Momentum_shift_pre'] = combined_df['Player2_Momentum'].shift(-1).fillna(0)
    return combined_df

def LR_data_process(file_path, max_shift):
    combined_df = read_and_combine_data(file_path)
    # 在最后一列加上计算好的两个玩家的momentum
    combined_df = append_momentum(combined_df, max_shift)

    # 动态生成特征列名
    feature_columns = [f'Player1_Momentum_shift_{i}' for i in range(1, max_shift)]
    features = combined_df[feature_columns]
    labels = combined_df['Player1_Momentum'].shift(max_shift).fillna(0)

    return features, labels

class mLinearRegression:
    def __init__(self, learning_rate=0.15, num_iterations=2100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()
        for i in range(self.num_iterations):
            for j in range(X.shape[0]):
                predictions = np.dot(X[j], self.weights) + self.bias
                errors = predictions - y[j]
                gradient = np.dot(X[j].T, errors) / X.shape[1]
                bias_gradient = np.sum(errors) / X.shape[1]
                self.weights -= self.learning_rate * gradient
                self.bias -= self.learning_rate * bias_gradient
        # for _ in range(self.num_iterations):
        #     for j in range(X.shape[0]):
        #         # 选择随机样本索引
        #         random_index = j
        #         X_sample = X[random_index]
        #         y_sample = y[random_index]
        #
        #         # 计算预测值和误差
        #         prediction = np.dot(X_sample, self.weights) + self.bias
        #         error = prediction - y_sample
        #
        #         # 计算梯度
        #         gradient = X_sample * error
        #         bias_gradient = error
        #
        #         # 更新权重和偏差
        #         self.weights -= self.learning_rate * gradient
        #         self.bias -= self.learning_rate * bias_gradient
        # return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # print(len(X))
        # print(len(self.weights))
        return np.dot(X, self.weights) + self.bias

class MCMLR(MCModel):
    def __init__(self):
        super().__init__()
        self.lr_model = mLinearRegression()
    def train(self, X, y):
        self.lr_model.fit(X, y)
    def predict(self, X):
        return self.lr_model.predict(X)
    def getModelParam(self):
        pass

def rolling_prediction(model, initial_values, steps):
    predictions = []
    input_values = initial_values.copy()
    x = input_values[:steps]
    for i in range(len(input_values)-steps):
        pre = model.predict(input_values[i:i+steps]).reshape(-1)
        predictions.append(pre)
        x = np.append(x, pre, axis=0)
    # for _ in range(steps):
    #     input_array = np.array(input_values)
    #     # 使用模型进行预测
    #     # print('input array',input_array)
    #     predicted_value = model.predict(input_array)
    #     # 更新输入值：移除最旧的值，加入最新的预测值
    #     input_values = np.roll(input_values, shift=-1)
    #     input_values[-1] = predicted_value
    #     # print(input_values)
    #     # input('stop')
    #     # 存储预测结果
    #     predictions.append(predicted_value)
    # for _ in range(len(initial_values)):
    #     prediction = model.predict(input_values)
    return predictions

def get_initial_momentum_values(file_path, max_shift):
    combined_df = read_and_combine_data([file_path])
    combined_df = append_momentum(combined_df, max_shift) # 这个时候类型是'pandas.core.series.Series'
    initial_values = combined_df['Player1_Momentum_shift_1'].head(max_shift).tolist()# 转化类型成为list，截取前100项
    return initial_values

def LR_main():
# Step1 训练模型
    model = MCMLR()
    # X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'], 100)
    # X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], 100)
    # X_train = np.array(X_train.values.tolist())
    # X_test = np.array(X_test.values.tolist())
    # y_train = np.array(y_train.values.tolist())
    # y_test = np.array(y_test.values.tolist())
    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)
    # model.train(X_train, y_train)
    train_list = mDR.getList('../../statics/training/session_train.csv')
    test_list = mDR.getList('../../statics/training/session_test.csv')
    y_train, p2m = mDR.getMomentum(train_list)
    y_test, p2m = mDR.getMomentum(test_list)

    train_len = len(y_train)
    test_len = len(y_test)
    changdu = 100

    X_train = []
    for i in range(train_len-changdu-1):
        single_x = y_train[i:i+changdu]
        X_train.append(single_x)
    X_test = []
    for i in range(test_len-changdu-1):
        single_x = y_test[i:i+changdu]
        X_test.append(single_x)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # model.train(X_train, y_train[changdu+1:])

# Step2 选取真实值的前100个，存为一个长度为100的list类型变量initial_values
    # initial values是一个长度为100的list
    # initial_values = get_initial_momentum_values('../../statics/training/session_test.csv', 99)
# Step3 进行rolling prediction，传入模型(用于predict)、initial_value、steps
#     steps = len(X_test)  # 假设想要预测整个测试集的长度
    model = model.load('../model_params/lr_roll_model.pkl')
    predictions = rolling_prediction(model, y_test, changdu)

    r2 = r2_score(y_test[100:], predictions)
    y_test = y_test[100:]
    # predictions = predictions[100:]
    # y_test = (y_test>0.5).astype(int)
    # predictions = np.array(predictions)
    # predictions = (predictions>0.5).astype(int)
    print(f'R^2 Score: {r2}')
    plt.figure()
    plt.plot(predictions, label='Predict')
    plt.plot(y_test, label='True')
    plt.legend()
    plt.show()
    print('R^2 Score:', model.r2_score(y_test, predictions))
    print('loss:', model.MSELoss(y_test, predictions))
    accuracy, precision, recall, f1 = model.judge(y_test, predictions)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # model.save('../model_params/lr_roll_model.pkl')

if __name__ == '__main__':
    LR_main()