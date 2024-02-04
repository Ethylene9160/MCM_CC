import numpy as np
<<<<<<< HEAD
from sklearn.metrics import r2_score
=======
import pickle
>>>>>>> b54530f66b90316bd0e700b5baccd452ee51b726
class MCModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def getModelParam(self):
        pass

    def MSELoss(self, y, y_pred):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return np.mean(np.square(y - y_pred))

<<<<<<< HEAD
    def r2(self, y, y_pred):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return r2_score(y, y_pred)
=======
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
>>>>>>> b54530f66b90316bd0e700b5baccd452ee51b726

    def judge(self, y, y_pred):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        y = (y>0.5).astype(int)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        y_pred = (y_pred>0.5).astype(int)
        TP = np.sum(np.logical_and(y_pred == 1, y == 1))
        FP = np.sum(np.logical_and(y_pred == 1, y == 0))
        FN = np.sum(np.logical_and(y_pred == 0, y == 1))
        accuracy = np.mean(np.equal(y, y_pred))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1
