from model.MCMLR import MCMLR, mLinearRegression
from model.MCMSVM import MSMSVM
from model.MCMSVM import MSMSVM as MCMSVM, SVM_data_process
from model.MCMMLP import MCMMLP, keys, CPP_Storage

import python.data_reader as mDR
import matplotlib.pyplot as plt

def visulaze_judgement(model, y, y_pred):
    print('loss: ', model.MSELoss(y, y_pred))
    print('R2 score: ', model.r2_score(y, y_pred))
    accuracy, precision, recall, f1 = model.judge(y, y_pred)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)


if __name__ == '__main__':
    X_test_ju, y_test_ju = SVM_data_process(['../statics/training/session_test.csv'], 5)
    # MLR model
    svm_model = MCMSVM()
    svm_model = svm_model.load('model_params/svm_model.pkl')
    svm_pred = svm_model.predict(X_test_ju)

    test_player_list = mDR.getList('../statics/training/session_test.csv')
    X_test_ethy, y_test_ethy = mDR.getXY(test_player_list, keys, 5)
    mlp_model = MCMMLP()
    mlp_model = mlp_model.load('model_params/mlp_model.pkl')
    mlp_pred = mlp_model.predict(X_test_ethy)

    lr_model = MCMLR()
    lr_model = lr_model.load('model_params/lr_model.pkl')
    lr_pred = lr_model.predict(X_test_ju)

    print('SVM results:')
    visulaze_judgement(svm_model, y_test_ju, svm_pred)

    print('MLP results:')
    visulaze_judgement(mlp_model, y_test_ethy, mlp_pred)

    print('LR results:')
    visulaze_judgement(lr_model, y_test_ju, lr_pred)

    # draw pic.
    plt.figure(figsize=(8, 5))
    plt.plot(y_test_ju, label='True')
    plt.plot(svm_pred, label='SVM Predict')
    plt.plot(mlp_pred, label='MLP Predict')
    plt.plot(lr_pred, label='LR Predict')
    plt.legend()
    plt.show()


