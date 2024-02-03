import numpy as np
class MCModel:
    def __init__(self):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def getModelParam(self):
        pass

    def MSELoss(self, y, y_pred):
        return np.mean(np.square(y - y_pred))

    def accuracy(self, y, y_pred):
        return np.mean(np.equal(y, y_pred))

    def F1_score(self, y, y_pred):
        TP = np.sum(np.logical_and(y_pred == 1, y == 1))
        FP = np.sum(np.logical_and(y_pred == 1, y == 0))
        FN = np.sum(np.logical_and(y_pred == 0, y == 1))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 * precision * recall / (precision + recall)

    def precision(self, y, y_pred):
        TP = np.sum(np.logical_and(y_pred == 1, y == 1))
        FP = np.sum(np.logical_and(y_pred == 1, y == 0))
        return TP / (TP + FP)
