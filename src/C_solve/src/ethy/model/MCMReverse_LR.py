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

    # 将momentum趋势添加到DataFrame中
    combined_df['Player1_Momentum'] = p1m
    combined_df['Player2_Momentum'] = p2m

    for i in range(1, max_shift):
        combined_df[f'Player1_Momentum_shift_{i}'] = combined_df['Player1_Momentum'].shift(i).fillna(0)
        combined_df[f'Player2_Momentum_shift_{i}'] = combined_df['Player2_Momentum'].shift(i).fillna(0)

    combined_df['Player1_Momentum_shift_pre'] = combined_df['Player1_Momentum'].shift(-1).fillna(0)
    combined_df['Player2_Momentum_shift_pre'] = combined_df['Player2_Momentum'].shift(-1).fillna(0)

    # 在测试集的最后一列加上一列，前5个值是真实的momentum，后面全部是0
    combined_df['Player1_Momentum_shift_new'] = 0
    combined_df.loc[:4, 'Player1_Momentum_shift_new'] = combined_df['Player1_Momentum'].head(5)

    return combined_df


def LR_data_process(file_path, is_test=False):
    combined_df = read_and_combine_data(file_path)

    # Provide the missing 'max_shift' argument
    combined_df = append_momentum(combined_df, max_shift=5)

    if is_test:
        # 仅在测试集中执行此操作
        combined_df['Test_Momentum'] = 0  # 初始化为0
        combined_df.loc[:4, 'Test_Momentum'] = combined_df.loc[:4, 'Player1_Momentum']  # 假设使用Player1的momentum
    features = combined_df[['Player1_Momentum']]
    labels = combined_df['Player1_Momentum']
    return features, labels


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


def LR_main():
    model = MCMLR()
    # 假设模型已经被训练并保存了
    model = model.load('../model_params/lr_model.pkl')

    # 处理训练数据
    # X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'])

    # 处理测试数据，标记为测试集以触发特定逻辑
    X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], is_test=True)

    # 开始逐步预测，并更新Test_Momentum列
    for i in range(5, len(X_test)):
        # 使用前5个真实momentum进行预测
        pred = model.predict(X_test.iloc[i - 5:i]).iloc[-1]
        X_test.iloc[i, X_test.columns.get_loc('Test_Momentum')] = pred  # 更新Test_Momentum列

    predictions = model.predict(X_test['Test_Momentum'])
    r2 = r2_score(y_test, predictions)
    print(f'R^2 Score: {r2}')

    plt.figure()
    plt.plot(predictions, label='Predict')
    plt.plot(y_test, label='True')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    LR_main()