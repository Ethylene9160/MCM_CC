import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import pandas as pd
import python.data_reader as mDR

# 假设data_reader和MCModel类已经正确实现，这里不再展示其代码

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

def LR_data_process(file_path, max_shift):
    combined_df = read_and_combine_data(file_path)
    # 在最后一列加上计算好的两个玩家的momentum
    combined_df = append_momentum(combined_df, max_shift)
    # 准备训练数据 -> 当前的特征+之前五个的momentum
    features = combined_df[['Player1_Momentum_shift_1', 'Player1_Momentum_shift_2', 'Player1_Momentum_shift_3', 'Player1_Momentum_shift_4']]
    labels = combined_df['Player1_Momentum']
    return features, labels

def rolling_prediction(test_data, model, initial_values, steps):
    # 初始化用于存储预测结果的数组
    predictions = []
    # 使用测试集中的前5个真实momentum值作为初始输入
    input_values = initial_values.copy()

    for _ in range(steps):
        # 将当前的输入值转换为模型预期的格式
        input_array = np.array([input_values])  # 将input_values包装为二维数组
        # 使用模型进行预测
        predicted_value = model.predict(input_array)[0]
        # 更新输入值：移除最旧的值，加入最新的预测值
        input_values = np.roll(input_values, shift=-1)
        input_values[-1] = predicted_value[0]
        # 存储预测结果
        predictions.append(predicted_value)

    return predictions


class mLinearRegression:
    def __init__(self, learning_rate=0.15, num_iterations=2100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()
        for i in range(self.num_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y
            gradient = np.dot(X.T, errors) / X.shape[0]
            bias_gradient = np.sum(errors) / X.shape[0]
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_gradient
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class MCMLR:
    def __init__(self):
        self.lr_model = mLinearRegression()

    def train(self, X, y):
        self.lr_model.fit(X, y)

    def predict(self, X):
        return self.lr_model.predict(X)

# def get_initial_momentum_values(file_path, max_shift):
#     combined_df = read_and_combine_data([file_path])
#     combined_df = append_momentum(combined_df, max_shift)
#     initial_values = combined_df['Player1_Momentum'].head(max_shift).values
#     return initial_values
def get_initial_momentum_values(file_path, max_shift):
    combined_df = read_and_combine_data([file_path])
    combined_df = append_momentum(combined_df, max_shift)
    initial_values = combined_df[['Player1_Momentum_shift_1', 'Player1_Momentum_shift_2', 'Player1_Momentum_shift_3', 'Player1_Momentum_shift_4']].head(max_shift).values
    return initial_values


# def LR_main():
#     # 初始化模型
#     model = MCMLR()
#
#     # 读取、处理训练数据，并训练模型
#     X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'], 5)
#     model.train(X_train, y_train)
#
#     # 读取、处理测试数据
#     X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], 5)
#
#     # 从测试数据中获取前5个momentum值作为初始值
#     initial_values = get_initial_momentum_values('../../statics/training/session_test.csv', 5)
#
#     # 进行滚动预测
#     steps = len(X_test)  # 假设想要预测整个测试集的长度
#     rolling_predictions = rolling_prediction(X_test, model, initial_values, steps)
#
#     # 绘制预测结果
#     plt.figure()
#     plt.plot(rolling_predictions, label='Rolling Predicted')
#     plt.legend()
#     plt.show()
def LR_main():
    # 初始化模型
    model = MCMLR()

    # 读取、处理训练数据，并训练模型
    X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'], 5)
    model.train(X_train, y_train)

    # 读取、处理测试数据
    X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], 5)

    # 从测试数据中获取前5个momentum值作为初始值
    initial_values = get_initial_momentum_values('../../statics/training/session_test.csv', 5)

    # 进行滚动预测
    steps = len(X_test)  # 假设想要预测整个测试集的长度
    rolling_predictions = rolling_prediction(X_test, model, initial_values.flatten(), steps)

    # 将初始值和滚动预测值合并
    combined_predictions = np.concatenate((initial_values.flatten()[:5], rolling_predictions))

    # 绘制真实的momentum值和预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Momentum', color='blue')  # 真实的momentum值
    plt.plot(combined_predictions, label='Rolling Predicted', color='red', linestyle='--')  # 滚动预测值
    plt.title('True vs Predicted Momentum')
    plt.xlabel('Time Step')
    plt.ylabel('Momentum')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    LR_main()