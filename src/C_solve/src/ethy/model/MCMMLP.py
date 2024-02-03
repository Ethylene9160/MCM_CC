from ethy.model.MCModel import MCModel
import cppyy
import ctypes
import numpy as np

import cppyy
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

import python.data_reader as mDR

##############################
### change the path ples #####
##############################
cppyy.include('../../cpp/PY_MPL.h')

class MCMMLP(MCModel):
    def __init__(self, epoches=5000, lr=0.03, inputSize=2, hidden_layer_sizes=[3]):
        super().__init__()
        self.mn = cppyy.gbl.PY_MPL()
        hlv = cppyy.gbl.std.vector[int](hidden_layer_sizes)
        self.mn.initMPL(epoches, lr, inputSize, hlv)

    def reset(self, epoches=5000, lr=0.03, inputSize=2, hidden_layer_sizes=[3]):
        hlv = cppyy.gbl.std.vector[int](hidden_layer_sizes)
        self.mn.initMPL(epoches, lr, inputSize, hlv)

    def train(self, data, labels):
        # judge thether labels are NumPy array
        if isinstance(data, np.ndarray):
            data_list = data.tolist()
        else:
            data_list = data

        if isinstance(labels, np.ndarray):
            labels_list = labels.tolist()
        else:
            labels_list = labels

        # declare std::vector in python
        vector_vector_double = cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']]
        vector_double = cppyy.gbl.std.vector['double']

        # transverse inputs
        inputs = vector_vector_double()
        for sample in data_list:
            sample_vector = vector_double()
            for feature in sample:
                sample_vector.push_back(feature)
            inputs.push_back(sample_vector)

        # transverse labels
        labels_vector = vector_double()
        for label in labels_list:
            labels_vector.push_back(label)

        self.mn.train(inputs, labels_vector)

    def predict(self, data):
        # 如果data是一维列表或者numpy数组
        if isinstance(data, np.ndarray) and len(data.shape) == 1:
            return self._single_predict(data)
        pred = []
        # 判断data是不是一个numpy数组
        # pred = []
        # if isinstance(data, np.ndarray):
        #     data = data.tolist()
        # # 判断data是不是一个list
        # elif isinstance(data, list):
        #     data = np.array(data)
        # # 判断data是不是一个数字
        # else:
        #     return self._single_predict(data)
        for sample in data:
            pred.append(self._single_predict(sample))
        return pred

    def _single_predict(self, data):
        if isinstance(data, np.ndarray):
            test_data_list = data.tolist()
        else:
            test_data_list = data
        vector_double = cppyy.gbl.std.vector['double']
        test_input = vector_double()
        for sample in test_data_list:
            test_input.push_back(sample)
        return self.mn.predict_reg(test_input)

    def setLR_VOKE(self, LR_VOKE):
        self.mn.setLR_VOKE(LR_VOKE)

    def getLosses(self):
        losses = cppyy.gbl.std.vector['double']()
        losses = self.mn.getLosses()
        losses_py = [losses[i] for i in range(len(losses))]
        return losses_py

if __name__ == '__main__':
    # player_list = mDR.getList('../../statics/training/session_train.csv')
    # 加载CSV数据
    keys = [
                            'p1_points_won', 'p2_points_won', \
                            'p1_double_fault', 'p2_double_fault', \
                            'p1_distance_run', 'p2_distance_run', \
                            'p1_break_pt_won', 'p2_break_pt_won', \
                            'p1_break_pt_missed', 'p2_break_pt_missed', \
                            'p1_unf_err','p2_unf_err',\
                            'p1_sets', 'p2_sets', \
                            'p1_games', 'p2_games', \
                            'p1_score', 'p2_score', \

                            'p1_winner', 'p1_ace'
    ]
    train_player_list = mDR.getList('../../statics/training/session_train.csv')
    test_player_list = mDR.getList('../../statics/training/session_test.csv')
    X_train,y_train = mDR.getXY(train_player_list, keys, 5)
    p1m, p2m = mDR.getMomentum(train_player_list)
    X_test, y_test = mDR.getXY(test_player_list, keys, 5)

    model = MCMMLP(epoches=1000, lr=0.0005, inputSize=len(X_test[0]), hidden_layer_sizes=[3,8,4])
    print('MLP start training!')
    # print(len(X_train))
    # print(len(y_train))
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    # y_test = (y_test>0.5).astype(int)
    plt.figure(figsize=(8, 5))
    plt.plot(y_pred, label='predict')
    plt.plot(y_test, label='true')
    plt.legend()
    plt.show()
    print('MSEloss:', model.MSELoss(y_test, y_pred))
    accuracy, precision, recall, f1 = model.judge(y_test, y_pred)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    model.save('../model_params/model.pkl')
