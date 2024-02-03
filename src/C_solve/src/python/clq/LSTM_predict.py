import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import savemat
import LSTM_Method
import python.data_reader as mDR
import pandas as pd

data_path = '../statics/29splits/session1.csv'
data = {}
features = {}  # 元素为list
labels = {}
historical = 5
batch_size= 64
n_train = 3 ## !!!!!!!!!!!!!!!!!!!!!
input_size = 2
hidden_size = 100
output_size = 2
# train_features = torch.zeros(1,5, input_size)
# train_labels = torch.zeros(1,5,input_size)
# test_features = torch.zeros(1,5, input_size)
# test_labels =  torch.zeros(1,5, input_size)
loss = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
losses = []


try:
    for i in range(1, 6):
        data_path = f'../../statics/29splits/session{i}.csv'
        original_data = mDR.read_data(data_path)
        # 读取数据，存储在player_list中。
        # player_list是一个列表，其中每个元素都是一个字典，代表一个选手的数据。
        player_list = []
        for index, row in original_data.iterrows():
            player_data = row.to_dict()
            player_list.append(player_data)
        # 计算两者的momentum趋势。
        # 这个函数在data_reader.py中定义
        # getMonmentum函数返回两个列表，分别代表两个选手的momentum趋势。
        # 传入的参数是包含对手对战的字典信息的列表。
        p1m, p2m = mDR.getMomentum(player_list)
        p1m = torch.tensor(p1m).view(-1,1)
        p2m = torch.tensor(p2m).view(-1,1)
        pm = torch.cat((p1m,p2m),dim=1)
        if i not in data:
            data[i] = []
        data[i].append(pm)
    for i in range(1,len(data)+1):
        if i not in features:
            features[i] = []
            labels[i] = []
        # print(f"1:{i}")
        feature, label,seq = LSTM_Method.slide_windows(data[i][0], historical)
        # print(f"2:{i}")
        features[i].append(feature)
        labels[i].append(label)
        features[i] = torch.cat(features[i], dim=0)
        labels[i] = torch.cat(labels[i], dim=0)

    # print(features)
    train_features = features[1]
    train_labels = labels[1]
    test_features = features[n_train]
    test_labels = labels[n_train]
    for i in range(2,n_train):
        train_features = torch.cat((train_features,features[i]), dim=0)
        train_labels = torch.cat((train_labels,labels[i]), dim=0)
    for i in range(n_train+1,len(data)+1):
        test_features = torch.cat((test_features,features[i]), dim=0)
        test_labels = torch.cat((test_labels,labels[i]), dim=0)


    train_features, min1,max1 = LSTM_Method.nor_maxmin(train_features)
    train_labels, min2,max2 = LSTM_Method.nor_maxmin(train_labels)
    test_features,min3,max3 = LSTM_Method.nor_maxmin(test_features)
    test_labels,min4,max4 = LSTM_Method.nor_maxmin(test_labels)

    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)


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


    def train(model, train_iter, loss, epochs, lr):
        # def train(model, features, labels, loss, epochs, lr):
        global l
        trainer = torch.optim.Adam(model.parameters(), lr)
        for epoch in range(epochs):
            for X, y in train_iter:
                # for X, y in zip(features[epoch, :, :], labels[epoch, :, :]):

                l = loss(model(X), y)
                # l = ((model(X)-y) ** 2)/len(y)
                trainer.zero_grad()
                l.mean().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, ')
            print(f'loss: {l.item()}')

    #
    my_model = LSTMModel(input_size, hidden_size, output_size)
    train(my_model, train_iter, loss, num_epochs, 0.01)
    torch.save(my_model.state_dict(), 'model_m.pt')

    my_state = torch.load('model_m.pt')
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(my_state)

    with torch.no_grad():
        test_out = model(test_features[:1000])
        # print(test_out)
        test_loss = loss(test_out, test_labels[:1000])
    print("Test Loss:", test_loss.item())

    test_out = LSTM_Method.inve_nor(test_out, min4, max4)
    test_labels = LSTM_Method.inve_nor(test_labels, min4, max4)
    savemat('test_out_m.mat', {'test_out': test_out,'test_labels':test_labels[:1000]})
    # print(test_out, test_labels)
    # test_features = LSTM_Method.inve_nor(test_features, min3, max3)
except Exception as e:
    print(e)


