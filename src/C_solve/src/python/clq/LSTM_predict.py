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
historical = 5
batch_size= 64
n_train = 10 ## !!!!!!!!!!!!!!!!!!!!!
input_size = 2
hidden_size = 100
output_size = 2
loss = nn.MSELoss()
num_epochs = 20
losses = []






"=====================Model====================="
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3,
                            batch_first=True)  # batch_first是什么意思 ， 输入数据x的维度：(batch_size, sequence_length, input_size)->(batch_size,4,2)
        self.fc = nn.Linear(hidden_size, output_size)  # 预测未来1个点

    # 输出张量的形状 (batch_size, sequence_length, num_directions * hidden_size)，其中
    # batch_size 表示批处理大小，即每个时间步上输入的数据样本数。
    # sequence_length 表示时间步数，即序列的长度。
    # num_directions 表示 LSTM 层的方向性，通常为 1（单向）或 2（双向）。
    # hidden_size 表示 LSTM 单元的隐藏状态的维度。
    # 每个元素 (i, j, k) 表示第 i 个样本（批处理中的样本索引），在序列的第 j 个时间步上，第 k 个隐藏单元的值。这个输出张量包含了模型在每个时间步上的中间隐藏状态信息。
    def forward(self, x):
        out, _ = self.lstm(x)  # out = (batch_size,4,100)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出  -1放在y轴
        return out.view(out.size(0), -1, 2)  # 重新调整输出形状 ???????????????


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
        self.eval()
        with torch.no_grad():
            prediction = self(test_features)
        self.train()
        return prediction
def main():
    "=====================Data Processing====================="
    train_iter, test_features, test_labels, min1, max1, min2, max2, min3, max3, min4, max4 = lm.data_processing(n_train,historical,batch_size)
    "=====================Train====================="
    # my_model = LSTMModel(input_size, hidden_size, output_size)
    # my_model.train_model(train_iter, loss, num_epochs, 0.01)
    # torch.save(my_model.state_dict(), 'output/model_m.pt')

    "=====================Load Model====================="
    my_state = torch.load('output/model_m.pt')
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(my_state)

    "=====================Test====================="
    output_len = 200
    test_out = model.predict(test_features[:output_len])
    test_loss = loss(test_out, test_labels[:output_len])
    print("Test Loss:", test_loss.item())
    test_out = lm.inve_nor(test_out, min4, max4)
    test_labels = lm.inve_nor(test_labels, min4, max4)
    savemat('output/test_out_m.mat', {'test_out': test_out,'test_labels':test_labels[:output_len]})

if __name__ == '__main__':
    main()

