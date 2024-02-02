import torch


def slide_windows(data, historical, slide_step=1, in_start=0, n_out=1):
    # data: 输入的数据集，(N，dimension)
    # historical: 历史序列长度
    # n_out: 一个序列输出多少个点
    # slide_step: 每次窗口往前移动多少位
    # in_start: 从第一个数据点开始取
    N = data.size(0)
    dimension = data.size(1)
    seq_num = (N - historical // slide_step)  # 运用滑动窗口方法，把长度为N的序列分成seq_num个长度为time_step的序列
    features = torch.zeros(seq_num, historical, dimension)  # 新建features shape = (seq_num,time_step,2)
    labels = torch.zeros(seq_num, 1, dimension)
    for i in range(N):
        in_end = in_start + historical
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end <= N:
            # 训练数据以滑动步长slide_step截取
            features[i, :, :] = data[in_start:in_end]
            labels[i, :, :] = data[in_end].unsqueeze(0)
        in_start += slide_step  # 滑动步长为slide_step，即往前移一步
    return features, labels, seq_num


## 归一化
def nor_maxmin(my_tensor,min,max):
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
    features, labels, seq_num = slide_windows(data[:num_samples], historical)  # 先生成第一条轨迹的features,labels
    for i in range(1, match_num):
        new_features, new_labels, _ = slide_windows(data[i * num_samples:(i + 1) * num_samples],
                                                    historical)  # 每一条轨迹生成相应的features,labels  (序列总数,time_step,2),(序列总数,1,2)
        features = torch.cat((features, new_features), dim=0)  # 连接起来
        labels = torch.cat((labels, new_labels), dim=0)
    return features, labels,seq_num
