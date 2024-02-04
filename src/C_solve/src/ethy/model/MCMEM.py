import pickle

from MCMLR import MCMLR, mLinearRegression
from MCMSVM import MSMSVM
import clq.LSTM_Method
from MCMSVM import MSMSVM as MCMSVM, SVM_data_process
from MCMMLP import MCMMLP, keys, CPP_Storage
from clq.LSTM_predict import AttentionLayer, LSTMModel

import clq.LSTM_predict as lst
import clq.LSTM_Method as lm

import python.data_reader as mDR
import matplotlib.pyplot as plt
import torch
import numpy as np

from ethy.model.MCModel import MCModel


def visulaze_judgement(model, y, y_pred):
    loss = model.MSELoss(y, y_pred)
    r2_score = model.r2_score(y, y_pred)

    print('loss: ', loss)
    print('R2 score: ', r2_score)
    accuracy, precision, recall, f1 = model.judge(y, y_pred)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    return 1-loss+r2_score+accuracy+precision+recall+f1


class mStore:
    def __init__(self, prediction, svm, lr, lstm, w_svm, w_mlp, w_lr, w_lstm):
        self.prediction = prediction
        self.svm = svm
        self.lr = lr
        self.lstm = lstm
        self.w_svm = w_svm
        self.w_mlp = w_mlp
        self.w_lr = w_lr
        self.w_lstm = w_lstm

class MCMEM(MCModel):
    def __init__(self):
        super().__init__()
        self.base_path = '../../'
        self.path = self.base_path+'statics/training/session_test.csv'
        self.svm_model = MCMSVM()
        self.svm_model = self.svm_model.load(self.base_path+'ethy/model_params/svm_model.pkl')

        self.mlp_model = MCMMLP()
        self.mlp_model = self.mlp_model.load(self.base_path+'ethy/model_params/mlp_model.pkl')

        self.lr_model = MCMLR()
        self.lr_model = self.lr_model.load(self.base_path+'ethy/model_params/lr_model.pkl')


        my_state = torch.load(self.base_path+'clq/output/model_whole.pt')
        self.lst_model = LSTMModel(lst.input_size, lst.hidden_size, lst.output_size)
        self.lst_model.load_state_dict(my_state)

        self.X_test_ju = None
        self.y_test_ju = None
        self.X_test_ethy = None
        self.y_test_ethy = None
        self.X_test_cai = None
        self.y_test_cai = None

        self.svm_pred = None
        self.mlp_pred = None
        self.lr_pred = None
        self.lstm_pred = None

        self.w_svm = 0
        self.w_mlp = 0
        self.w_lr = 0
        self.w_lstm = 0

    def _judgement(self, model, y, y_pred):
        loss = model.MSELoss(y, y_pred)
        r2_score = model.r2_score(y, y_pred)

        # print('loss: ', loss)
        # print('R2 score: ', r2_score)
        accuracy, precision, recall, f1 = model.judge(y, y_pred)
        # print('accuracy:', accuracy)
        # print('precision:', precision)
        # print('recall:', recall)
        # print('f1:', f1)
        return 1 - loss + r2_score + accuracy + precision + recall + f1
    def set_path(self, path):
        self.path = path
    def train(self, X = 0, maxShift = 5):
        self.X_test_ju, self.y_test_ju = SVM_data_process([self.path],maxShift)
        self.X_test_ethy, self.y_test_ethy = mDR.getXY(mDR.getList(self.path), keys, maxShift)
        train_iter, test_features, test_labels, min1, max1, min2, max2, min3, max3, min4, max4 = lm.data_processing(
            lst.n_train,
            lst.historical,
            lst.batch_size,
            lst.total,
            lst.header_list,self.base_path)
        self.y_test_cai = test_labels[:, 0, 0].reshape(-1, 1)

        self.svm_pred = self.svm_model.predict(self.X_test_ju)
        self.mlp_pred = self.mlp_model.predict(self.X_test_ethy)
        self.lr_pred = self.lr_model.predict(self.X_test_ju)
        # self.lstm_pred = self.lstm_model.predict(test_features)

        lstm_pred = self.lst_model.predict(test_features)
        lstm_pred = lm.inve_nor(lstm_pred, min4, max4)
        test_labels = lm.inve_nor(test_labels, min4, max4)
        # test_out = lm.inve_nor(test_out, min4, max4)
        # test_labels = lm.inve_nor(test_labels, min4, max4)
        # print('test_out:',test_out)
        # 变成一个2*888的np数组

        self.lstm_pred = lstm_pred[:, 0, 0].reshape(-1, 1)
        self.y_test_cai = test_labels[:, 0, 0].reshape(-1, 1)
        self.w_svm = self._judgement(self.svm_model, self.y_test_ju, self.svm_pred)
        self.w_mlp = self._judgement(self.mlp_model, self.y_test_ethy, self.mlp_pred)
        self.w_lr = self._judgement(self.lr_model, self.y_test_ju, self.lr_pred)
        self.w_lstm = self._judgement(self.lst_model, self.y_test_cai, self.lstm_pred)
        self.w_svm/=1.11
        whole = np.exp(self.w_svm)+np.exp(self.w_mlp)+np.exp(self.w_lr)+np.exp(self.w_lstm)
        self.w_svm = np.exp(self.w_svm)/whole
        self.w_mlp = np.exp(self.w_mlp)/whole
        self.w_lr = np.exp(self.w_lr)/whole
        self.w_lstm = np.exp(self.w_lstm)/whole
        self.prediction = np.zeros(len(self.svm_pred))
        for i in range(len(self.prediction)):
            self.prediction[i] = self.w_svm * self.svm_pred[i] + self.w_mlp * self.mlp_pred[i] + self.w_lr * self.lr_pred[i] + self.w_lstm * self.lstm_pred[i]


    def predict(self, X = 0):
        return self.prediction




    def save(self, path):
        self.mlp_model.save(path+'s')


        str = mStore(self.prediction, self.svm_model,self.lr_model,self.lst_model,self.w_svm, self.w_mlp,self.w_lr,self.w_lstm)
        with open(path, 'wb') as file:
            pickle.dump(str, file)

    def load(self, path):
        self.mlp_model = MCMMLP()
        self.mlp_model = self.mlp_model.load(path+'s')
        with open(path, 'rb') as file:
            mstore = pickle.load(file)
        self.svm_model = mstore.svm
        self.lr_model = mstore.lr
        self.lst_model = mstore.lstm
        self.w_svm = mstore.w_svm
        self.w_mlp = mstore.w_mlp
        self.w_lr = mstore.w_lr
        self.w_lstm = mstore.w_lstm
        self.prediction = mstore.prediction
        return self



if __name__ == '__main__':
    X_test_ju, y_test_ju = SVM_data_process(['../../statics/training/session_test.csv'], 5)
    mcmem_model = MCMEM()
    mcmem_model.train(maxShift=5)
    mcmem_model = mcmem_model.load('../../ethy/model_params/em_model.pkl')
    predictions = mcmem_model.predict(X_test_ju)
    loss = mcmem_model.MSELoss(y_test_ju, predictions)
    print(f'Loss: {loss}')
    r2 = mcmem_model.r2_score(y_test_ju, predictions)
    print(f'R^2 Score: {r2}')

    accuracy, precision, recall, f1 = mcmem_model.judge(y_test_ju, predictions)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)


    #draw pic.
    plt.figure(figsize=(8, 5))
    plt.plot(y_test_ju, label='True')
    plt.plot(predictions, label='Predict')
    plt.legend()
    plt.show()

    # mcmem_model.save('../../ethy/model_params/em_model.pkl')



