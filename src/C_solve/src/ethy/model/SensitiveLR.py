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
    return combined_df

def LR_data_process(file_path, max_shift):
    combined_df = read_and_combine_data(file_path)
    # 在最后一列加上计算好的两个玩家的momentum
    combined_df = append_momentum(combined_df, max_shift)
    # 准备训练数据 -> 当前的特征+之前五个的momentum
    features = combined_df[[
                            'Player1_Momentum_shift_1', \
                            'Player1_Momentum_shift_2', \
                            'Player1_Momentum_shift_3', \
                            'Player1_Momentum_shift_4']]
    labels = combined_df['Player1_Momentum']
    return features, labels

class mLinearRegression:
    def __init__(self, learning_rate=0.15, num_iterations=2100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        # self.weights = np.random.randn(X.shape[1])
        # self.bias = np.random.randn()
        fixed_value = 0.5  # 你可以根据需要选择其他值
        self.weights = np.full(X.shape[1], fixed_value)
        self.bias = fixed_value
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
    model = model.load('../model_params/lr_model.pkl')
    X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'], 5)
    X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], 5)

    # model.train(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f'R^2 Score: {r2}')
    plt.figure()
    plt.plot(predictions, label='Predict')
    plt.plot(y_test, label='True')
    plt.legend()
    plt.show()
    print('R^2 Score:', model.r2(y_test, predictions))
    print('loss:', model.MSELoss(y_test, predictions))
    accuracy, precision, recall, f1 = model.judge(y_test, predictions)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    model.save('../model_params/lr_model.pkl')


def sensitivity_analysis():
    learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    num_iterations = [500, 1000, 1500, 2000, 2500]

    # 初始化空列表来存储结果
    results = []

    for lr in learning_rates:
        for num_iter in num_iterations:
            model = MCMLR()
            model.lr_model = mLinearRegression(learning_rate=lr, num_iterations=num_iter)
            X_train, y_train = LR_data_process(['../../statics/training/session_train.csv'], 5)
            X_test, y_test = LR_data_process(['../../statics/training/session_test.csv'], 5)

            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)

            results.append((lr, num_iter, r2))

    # 将结果转换为DataFrame以便于分析和可视化
    results_df = pd.DataFrame(results, columns=['Learning Rate', 'Num Iterations', 'R^2 Score'])

    # 可视化结果
    for lr in learning_rates:
        subset = results_df[results_df['Learning Rate'] == lr]
        plt.plot(subset['Num Iterations'], subset['R^2 Score'], label=f'lr={lr}')
        plt.legend(loc='lower right')
    plt.xlabel('Number of Iterations')
    plt.ylabel('R^2 Score')
    plt.legend()
    plt.title('LR Sensitivity Analysis: Learning Rate and Num Iterations')
    plt.legend(loc='lower right')
    plt.savefig('LR_Sensitivity_Analysis.eps')
    plt.show()

if __name__ == '__main__':
    # LR_main()
    sensitivity_analysis()