import python.data_reader as mDR

data = {}
for i in range(1,6):
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
    if i not in data:
        data[i] = []
    data[i].append([p1m,p2m])

