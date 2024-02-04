import numpy as np
import pickle
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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

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

    def r2_score(self, y, y_pred):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y)))