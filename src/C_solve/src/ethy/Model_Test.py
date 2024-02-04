from model.MCMLR import MCMLR, mLinearRegression
from model.MCMSVM import MSMSVM
import clq.LSTM_Method
from model.MCMSVM import MSMSVM as MCMSVM, SVM_data_process
from model.MCMMLP import MCMMLP, keys, CPP_Storage
from clq.LSTM_predict import AttentionLayer, LSTMModel

import clq.LSTM_predict as lst
import clq.LSTM_Method as lm

import python.data_reader as mDR
import matplotlib.pyplot as plt
import torch
import numpy as np

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

    "=====================Data Processing====================="
    train_iter, test_features, test_labels, min1, max1, min2, max2, min3, max3, min4, max4= lm.data_processing(lst.n_train,
                                                                                                                lst.historical,
                                                                                                                lst.batch_size,
                                                                                                                lst.total,
                                                                                                                lst.header_list)
    "=====================Train====================="
    # my_model = LSTMModel(input_size, hidden_size, output_size)
    # my_model.train_model(train_iter, loss, num_epochs, 0.01)
    # torch.save(my_model.state_dict(), 'output/model_m_8.pt')
    # plt.figure()
    # plt.title('label')
    # plt.plot(tmp, label='True')
    # plt.legend()
    # plt.show()
    "=====================Load Model====================="
    my_state = torch.load('../clq/output/model_whole.pt')
    model = LSTMModel(lst.input_size, lst.hidden_size, lst.output_size)
    model.load_state_dict(my_state)

    "=====================Test====================="
    # output_len = 888
    lstm_pred = model.predict(test_features)
    # test_loss = loss(test_out, test_labels)
    # print("Test Loss:", test_loss.item())
    lstm_pred = lm.inve_nor(lstm_pred, min4, max4)
    test_labels = lm.inve_nor(test_labels, min4, max4)
    # test_out = lm.inve_nor(test_out, min4, max4)
    # test_labels = lm.inve_nor(test_labels, min4, max4)
    # print('test_out:',test_out)
    # 变成一个2*888的np数组

    lstm_pred = lstm_pred[:, 0, 0].reshape(-1, 1)
    test_labels = test_labels[:,0,0].reshape(-1,1)
    #将test_out前九位数删除
    # test_out = test_out[9:]
    # 向test_out前补充9个0
    # test_out = np.concatenate((np.zeros((9, 1)), test_out), axis=0)
    # 将test_out维度降低，去掉最外层的维度
    # test_out = test_out[:, 1]
    # print('test_out:', test_out)
    # 然后提取出testout的第一列，赋值给它自己。


    print('SVM results:')
    w_svm = visulaze_judgement(svm_model, y_test_ju, svm_pred)
    w_svm /=1.2
    print('MLP results:')
    w_mlp = visulaze_judgement(mlp_model, y_test_ethy, mlp_pred)

    print('LR results:')
    w_lr = visulaze_judgement(lr_model, y_test_ju, lr_pred)

    print('LSTM results:')
    w_lstm = visulaze_judgement(model, test_labels, lstm_pred)
    w_whole = np.exp(w_svm)+np.exp(w_mlp)+np.exp(w_lr)+np.exp(w_lstm)


    w_svm = np.exp(w_svm)/w_whole
    w_mlp = np.exp(w_mlp)/w_whole
    w_lr = np.exp(w_lr)/w_whole
    w_lstm = np.exp(w_lstm)/w_whole
    print('w_svm:', w_svm)
    print('w_mlp:', w_mlp)
    print('w_lr:', w_lr)
    print('w_lstm:', w_lstm)
    prediction = np.zeros(len(svm_pred))
    for i in range(len(prediction)):
        prediction[i] = w_svm * svm_pred[i] + w_mlp * mlp_pred[i] + w_lr * lr_pred[i] + w_lstm * lstm_pred[i]
    # prediction = w_svm*svm_pred + w_mlp*mlp_pred + w_lr*lr_pred + w_lstm*test_out
    print('final result:')
    print('loss:', lr_model.MSELoss(y_test_ju, prediction))
    print('r2:', lr_model.r2_score(y_test_ju, prediction))
    accuracy, precision, recall, f1 = lr_model.judge(y_test_ju, prediction)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    # draw pic.
    plt.figure(figsize=(8, 5))
    plt.plot(y_test_ju, label='True')
    # plt.plot(y_test_ju, label='True2')
    plt.plot(svm_pred, label='SVM Predict')
    plt.plot(mlp_pred, label='MLP Predict')
    plt.plot(lr_pred, label='LR Predict')
    plt.plot(lstm_pred, label='LSTM Predict')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(y_test_ju, label='True')
    plt.plot(prediction, label='Final Predict')
    plt.legend()
    plt.show()


