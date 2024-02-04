from ethy.model.MCModel import MCModel
import cppyy
import ctypes
import numpy as np
import pickle

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
vector_double = cppyy.gbl.std.vector['double']
vector_vector_double = cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']]
vector_vector_vector_double = cppyy.gbl.std.vector[cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']]]

class CPP_Storage:
    def __init__(self, LR_VOKE, losses, epoches, lr, h, w, b):
        self.LR_VOKE = LR_VOKE
        self.losses = losses
        self.epoches = epoches
        self.lr = lr
        self.h = h
        self.w = w
        self.b = b

    def to_native_types(self):
        # Convert C++ std::vector<double> to Python list
        self.losses = self.to_list(self.losses)

        # Convert C++ std::vector<std::vector<double>> to nested Python lists
        self.h = self.to_list_list(self.h)
        self.w = self.to_list_list_list(self.w)
        self.b = self.to_list_list(self.b)

    def from_native_types(self):
        # Convert Python list back to C++ std::vector<double>
        self.losses = self.to_vector_double(self.losses)  # Assuming you have a method to create std::vector<double>

        # Convert nested Python lists back to C++ std::vector<std::vector<double>>
        self.h = self.to_vector_vector_double(self.h)
        self.w = self.to_vector_vector_vector_double(self.w)
        print('w',self.w)
        self.b = self.to_vector_vector_double(self.b)

    def save(self, file_path):
        # Convert to native types before saving
        self.to_native_types()
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            loaded_storage = pickle.load(file)
            # Convert back to C++ types after loading
            loaded_storage.from_native_types()
            return loaded_storage

    # 一维float数组到一维C++ std::vector
    def to_vector_double(self, data):
        vector_double = cppyy.gbl.std.vector['double']
        vec = vector_double()
        for d in data:
            vec.push_back(d)
        return vec

    # 二维float数组到二维C++ std::vector
    def to_vector_vector_double(self, data):
        vector_vector_double = cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']]
        vec = vector_vector_double()
        for d in data:
            vec.push_back(self.to_vector_double(d))
        return vec

    # 三维float数组到三维C++ std::vector
    def to_vector_vector_vector_double(self, data):
        vector_vector_vector_double = cppyy.gbl.std.vector[cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']]]
        vec = vector_vector_vector_double()
        for d in data:
            vec.push_back(self.to_vector_vector_double(d))
        return vec

    # 一维C++ std::vector到一维float数组
    def to_list(self, vec):
        return [vec[i] for i in range(len(vec))]

    # 二维C++ std::vector到二维float数组
    def to_list_list(self, vec):
        return [self.to_list(vec[i]) for i in range(len(vec))]

    def to_list_list_list(self, vec):
        return [self.to_list_list(vec[i]) for i in range(len(vec))]





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

    def save(self, path):
        storage = CPP_Storage(self.mn.getLR_VOKE(), self.mn.getLosses(), self.mn.getEpoches(), self.mn.getLR(), self.mn.getH(), self.mn.getW(), self.mn.getB())
        storage.save(path)

    def load(self, path):
        print('start load')
        # with open(path, 'rb') as f:
        #     storage = pickle.load(f)
        storage = CPP_Storage.load(path)
        print('end load')
        # storage.from_native_types()
        # self.mn.initMPL(storage.epoches, storage.lr, storage.w, storage.h)
        self.mn.load(storage.epoches, storage.lr, storage.w, storage.b, storage.h, storage.losses, storage.LR_VOKE)
        return self
        # self.mn.initMPL(storage.epoches, storage.lr, storage.w, storage.h)
        # self.mn.setLR_VOKE(storage.LR_VOKE)
        # self.mn.setLosses(storage.losses)
        # self.mn.setW(storage.w)
        # self.mn.setB(storage.b)

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

    model = MCMMLP(epoches=20000, lr=0.00035, inputSize=len(X_test[0]), hidden_layer_sizes=[10, 8])
    model.setLR_VOKE(200)
    print('MLP start training!')

    model.train(X_train, y_train)
    # load模型
    # model = model.load('../model_params/mlp_model.pkl')
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
