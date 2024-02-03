import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import savemat
import LSTM_Method as lm
import python.data_reader as mDR
import pandas as pd

data = {}
features = {}  # 元素为list
labels = {}
historical = 8
batch_size= 64
n_train = 10 ## !!!!!!!!!!!!!!!!!!!!!
hidden_size = 100
output_size = 2
loss = nn.MSELoss()
num_epochs = 50
total = 15
losses = []
# model_m_8
# header_list=[]
#  add2
# header_list = ['point_victor','game_victor','set_victor','p1_points_won', 'p2_points_won']
# add3
header_list = ['point_victor','game_victor','set_victor']
input_size = 2+len(header_list)






"=====================Model====================="

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attn_weights: (batch_size, seq_length, 1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        # context_vector: (batch_size, hidden_size)
        return context_vector

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)  # 因为是双向LSTM
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 注意力层输出的上下文向量大小与LSTM隐藏层大小一致

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size * 2)
        context_vector = self.attention(lstm_out)  # context_vector: (batch_size, hidden_size * 2)
        out = self.fc(context_vector)  # out: (batch_size, output_size)
        return out.view(out.size(0), -1, 2)


    def train_model(self, train_iter, loss, epochs, lr):
        # def train(model, features, labels, loss, epochs, lr):
        # super().train(mode=True)
        global l
        trainer = torch.optim.Adam(self.parameters(), lr)
        for epoch in range(epochs):
            for X, y in train_iter:
                # for X, y in zip(features[epoch, :, :], labels[epoch, :, :]):
                l = loss(self(X), y)
                # l = ((model(X)-y) ** 2)/len(y)
                trainer.zero_grad()
                l.mean().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, ')
            print(f'loss: {l.item()}')

    def predict(self, test_features):
        # self.eval()  # 设置模型为评估模式
        # predictions = []  # 用于存储所有的预测值

        # with torch.no_grad():  # 不计算梯度，减少计算和内存消耗
        #     current_features = test_features
        #     for _ in range(n_steps):
        #         prediction = self(current_features)  # 进行一次预测
        #          # 保存预测结果
        #
        #         # 将预测值添加到当前特征的末尾，并移除最前面的值以保持特征长度不变
        #         # 假设 current_features 和 prediction 都是 [batch_size, feature_size] 的形状
        #         # 这里假设沿着第二个维度（特征维度）进行操作
        #         current_features = torch.cat((current_features[:, 1:,:], prediction), dim=1)
        #
        # self.train()  # 将模型设置回训练模式
        # return prediction  # 返
        self.eval()
        with torch.no_grad():
            prediction = self(test_features)
        self.train()
        return prediction

    def MSELoss(self, y, y_pred):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return np.mean(np.square(y - y_pred))

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
def main():
    "=====================Data Processing====================="
    train_iter, test_features, test_labels, min1, max1, min2, max2, min3, max3, min4, max4 = lm.data_processing(n_train,historical,batch_size,total,header_list)
    "=====================Train====================="
    # my_model = LSTMModel(input_size, hidden_size, output_size)
    # my_model.train_model(train_iter, loss, num_epochs, 0.01)
    # torch.save(my_model.state_dict(), 'output/model_m_8.pt')

    "=====================Load Model====================="
    my_state = torch.load('output/model_add3.pt')
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(my_state)

    "=====================Test====================="
    # output_len = 888
    test_out = model.predict(test_features)
    test_loss = loss(test_out, test_labels)
    print("Test Loss:", test_loss.item())
    # test_out = lm.inve_nor(test_out, min4, max4)
    # test_labels = lm.inve_nor(test_labels, min4, max4)
    accuracy, precision, recall, f1 = model.judge(test_labels, test_out)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    savemat('output/test_out_m.mat', {'test_out': test_out,'test_labels':test_labels})

if __name__ == '__main__':
    main()

