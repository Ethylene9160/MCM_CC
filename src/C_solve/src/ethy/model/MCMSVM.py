from ethy.model.MCModel import MCModel

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

class MSMSVM(MCModel):
    def __init__(self):
        # 训练SVM模型
        super().__init__()
        self.svm_model = SVR()

    def train(self, X, y):
        self.svm_model.fit(X, y)

    def predict(self, X):
        return self.svm_model.predict(X)

    def getModelParam(self):
        pass

def SVM_main():
    # 使用方法：
    model = MSMSVM()
    # 读取数据
    X_train, y_train = SVM_data_process(['../../statics/training/session_train.csv'], 5)
    X_test, y_test = SVM_data_process(['../../statics/training/session_test.csv'], 5)

    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f'R^2 Score: {r2}')
    #draw pic.
    plt.figure()
    plt.plot(y_test, label='True')
    plt.plot(predictions, label='Predict')
    plt.legend()
    plt.show()

    print('loss:', model.MSELoss(y_test, predictions))
    print('accuracy:', model.accuracy(y_test, predictions))
    print('precision:', model.precision(y_test, predictions))
    print('recall:', model.recall(y_test, predictions))
    print('f1:', model.F1_score(y_test, predictions))

if __name__ == '__main__':
    SVM_main()
    # pass
