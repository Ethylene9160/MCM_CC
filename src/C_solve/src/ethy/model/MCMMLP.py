# from ethy.model.MCModel import MCModel
import cppyy
import ctypes
import numpy as np

import cppyy
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split



##############################
### change the path ples #####
##############################
cppyy.include('../../cpp/PY_MPL.h')

class MCMMLP():
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
        return self.mn.predict(test_input)

    def setLR_VOKE(self, LR_VOKE):
        self.mn.setLR_VOKE(LR_VOKE)

    def getLosses(self):
        losses = cppyy.gbl.std.vector['double']()
        losses = self.mn.getLosses()
        losses_py = [losses[i] for i in range(len(losses))]
        return losses_py
class MyMPL:
    def __init__(self, epoches=5000, lr=0.03, inputSize=2, hidden_layer_sizes=[3]):
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
        if isinstance(data, np.ndarray):
            test_data_list = data.tolist()
        else:
            test_data_list = data
        vector_double = cppyy.gbl.std.vector['double']
        test_input = vector_double()
        for sample in test_data_list:
            test_input.push_back(sample)
        return self.mn.predict(test_input)

    def setLR_VOKE(self, LR_VOKE):
        self.mn.setLR_VOKE(LR_VOKE)

    def getLosses(self):
        losses = cppyy.gbl.std.vector['double']()
        losses = self.mn.getLosses()
        losses_py = [losses[i] for i in range(len(losses))]
        return losses_py

if __name__ == '__main__':
    # 生成数据
    np.random.seed(0)
    X, y = np.random.random((100, 2)), np.random.random((100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train, y_train)
    # 训练模型
    model = MCMMLP(epoches=5000, lr=0.01, inputSize=2, hidden_layer_sizes=[3, 3])
    model.train(X_train, y_train)

    # 预测

    # y_pred=[]
    # for X in X_test:
    #     y_pred.append(model.predict(X))
    # print(y_pred)
    y_pred = model.predict(X_test)

    # 画出损失函数变化趋势
    losses = model.getLosses()
    plt.plot(losses)
    plt.show()