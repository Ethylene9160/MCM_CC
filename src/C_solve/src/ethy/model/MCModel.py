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