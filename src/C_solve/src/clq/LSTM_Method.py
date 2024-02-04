import torch
import python.data_reader as mDR
from torch.utils.data import TensorDataset, DataLoader

def slide_windows(data, historical, slide_step=1, in_start=0, n_out=1):
    # data: 输入的数据集，(N，dimension)
    # historical: 历史序列长度
    # n_out: 一个序列输出多少个点
    # slide_step: 每次窗口往前移动多少位
    # in_start: 从第一个数据点开始取
    N = data.size(0)
    dimension = data.size(1)
    # seq_num = (N - historical // slide_step)  # 运用滑动窗口方法，把长度为N的序列分成seq_num个长度为time_step的序列
    features = torch.zeros(N, historical, dimension)  # 新建features shape = (seq_num,time_step,2)
    labels = torch.zeros(N, 1, 2)
    for i in range(historical):
        a = torch.cat((torch.zeros(historical-i, dimension), data[:i,:]), dim=0)
        # print(a)
        features[i, :, :] = torch.cat((torch.zeros(historical-i, dimension), data[:i,:]), dim=0)
        labels[i, :, :] = data[i,:2].unsqueeze(0)
    for i in range(historical,N):
        in_end = in_start + historical
        out_end = in_end + n_out
        # print(in_end)

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end <= N:
            # 训练数据以滑动步长slide_step截取
            features[i, :, :] = data[in_start:in_end]
            labels[i, :, :] = data[in_end,:2].unsqueeze(0)
            # features[i, :, :] = data[in_start:in_end, :]
            # # 计算当前时刻与下一时刻的差值
            # labels[i, :, :] = data[in_end:out_end, :] - data[in_end - 1:in_end, :]
        in_start += slide_step  # 滑动步长为slide_step，即往前移一步
    return features, labels


## 归一化
def nor_maxmin(my_tensor):
    min = []
    max = []
    for i in range(my_tensor.size(2)):
        min.append(torch.min(my_tensor[:, :, i]))
        max.append(torch.max(my_tensor[:, :, i]))
        delta = max[i] - min[i]
        if delta != 0:
            my_tensor[:, :, i] = (my_tensor[:, :, i] - min[i]) / delta
    return my_tensor, min,max

## 反归一化
def inve_nor(my_tensor, min,max):
    for i in range(len(min)):
        my_tensor[:,:,i] = my_tensor[:,:,i]*(max[i]-min[i])+min[i]
    return my_tensor


def ger_feature_label(data, historical,num_samples,match_num):
    features, labels = slide_windows(data[:num_samples], historical)  # 先生成第一条轨迹的features,labels
    for i in range(1, match_num):
        new_features, new_labels= slide_windows(data[i * num_samples:(i + 1) * num_samples],
                                                    historical)  # 每一条轨迹生成相应的features,labels  (序列总数,time_step,2),(序列总数,1,2)
        features = torch.cat((features, new_features), dim=0)  # 连接起来
        labels = torch.cat((labels, new_labels), dim=0)
    return features, labels

def get_data(player_list,header):
    return_data = []
    for i in range(len(player_list)):
        return_data.append(player_list[i][f'{header}'])
    return return_data

def data_processing(n_train,historical,batch_size,total,header_list):
    data = {}
    features = {}  # 元素为list
    labels = {}
    for i in range(1, total):
        data_path = f'../statics/29splits/session{i}.csv'
        original_data = mDR.read_data(data_path)
        player_list = []
        for index, row in original_data.iterrows():
            player_data = row.to_dict()
            player_list.append(player_data)
        # 计算两者的momentum趋势。
        # 这个函数在data_reader.py中定义
        # getMonmentum函数返回两个列表，分别代表两个选手的momentum趋势。
        # 传入的参数是包含对手对战的字典信息的列表。
        p1m, p2m = mDR.getMomentum(player_list)
        # winner = get_data(player_list, 'point_victor')
        p1m = torch.tensor(p1m).view(-1, 1)
        p2m = torch.tensor(p2m).view(-1, 1)
        pm = torch.cat((p1m, p2m), dim=1)
        # winner = torch.tensor(winner).view(-1, 1)
        for header in header_list:
            p = get_data(player_list, header)
            p = torch.tensor(p).view(-1, 1)
            pm = torch.cat((pm,p), dim=1)
        if i not in data:
            data[i] = []
        data[i].append(pm)
    for i in range(1, len(data) + 1):
        if i not in features:
            features[i] = []
            labels[i] = []
        # print(f"1:{i}")
        feature, label = slide_windows(data[i][0], historical)
        # print(f"2:{i}")
        features[i].append(feature)
        labels[i].append(label)
        features[i] = torch.cat(features[i], dim=0)
        labels[i] = torch.cat(labels[i], dim=0)

    # a = 200
    # print(data[1][0][8+a])
    # # # print(features[1][:7])
    # print(labels[1][0+a])


    """===============分割测试集和数据集======================"""
        # print(features)
    train_features = features[1]
    train_labels = labels[1]
    test_features = features[n_train]
    test_labels = labels[n_train]
    for i in range(2, n_train):
        train_features = torch.cat((train_features, features[i]), dim=0)
        train_labels = torch.cat((train_labels, labels[i]), dim=0)
    for i in range(n_train + 1, len(data)):
        test_features = torch.cat((test_features, features[i]), dim=0)
        test_labels = torch.cat((test_labels, labels[i]), dim=0)

    train_features, min1, max1 = nor_maxmin(train_features)
    train_labels, min2, max2 = nor_maxmin(train_labels)
    test_features, min3, max3 = nor_maxmin(test_features)
    test_labels, min4, max4 = nor_maxmin(test_labels)

    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    return train_iter, test_features, test_labels, min1, max1, min2, max2, min3, max3, min4, max4