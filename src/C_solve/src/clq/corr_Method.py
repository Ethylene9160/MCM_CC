import numpy as np

'''标题为header的数据，返回list'''


def get_data(player_list, header):
    return_data = []
    for i in range(len(player_list)):
        return_data.append(player_list[i][f'{header}'])
    return return_data


'''random生成获胜情况'''


def simulate_random_match(num_points, serve_win_prob=0.65):
    # 随机生成每个得分点的获胜方，发球方获胜概率较高
    wins = np.random.rand(num_points) < serve_win_prob
    momentum_scores = np.cumsum(wins)  # 累积胜点作为简化的势头得分
    return momentum_scores


'''获取连续得分游程'''


def count_runs(sequence):
    runs = 1
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            runs += 1
    return runs


'''计算连续得分游程的期望和方差'''


def runs_stats(sequence):
    N = len(sequence)
    E = 1 + 0.5 * (N - 1)
    var = 0.25 * (N - 1)
    return E, var

'''计算z'''

def z_score(sequence):
    R = count_runs(sequence)
    E, var = runs_stats(sequence)
    print(E, var)
    print(R)
    return (R - E) / np.sqrt(var)


'''计算每一轮出错率'''
def error_rate(player_list,i):
    df = get_data(player_list,f'p{i}_double_fault')
    uf = get_data(player_list,f'p{i}_unf_err')
    net =  get_data(player_list,f'p{i}_net_pt')
    pm = get_data(player_list,f'p{i}_break_pt_missed')
    error = [df,uf,net,pm]
    error = np.array(error)
    err = []
    for i in range(len(error[1])):
        esum = np.sum(error[:,i])
        if esum > 0 :
            err.append(1)
        else:
            err.append(0)
    return err

def get_error_sum(player_list,i):
    df = get_data(player_list,f'p{i}_double_fault')
    uf = get_data(player_list,f'p{i}_unf_err')
    net =  get_data(player_list,f'p{i}_net_pt')
    pm = get_data(player_list,f'p{i}_break_pt_missed')
    error = [df,uf,net,pm]
    error = np.array(error)
    err = []
    for i in range(len(error[1])):
        esum = np.sum(error[:,:i])
        err.append(esum)
    return err
def get_server(player_list):
    serve = get_data(player_list,'server')
    serve = [1 if x == 1 else 0 for x in serve]
    return serve

def get_surprise(player_list,i):
    pw = get_data(player_list,f'p{i}_winner')
    npw = get_data(player_list,f'p{i}_net_pt_won')
    bp = get_data(player_list,f'p{i}_break_pt')
    bpw = get_data(player_list,f'p{i}_break_pt_won')
    sur = [pw,npw,bp,bpw]
    sur = np.array(sur)
    surprise = []
    for i in range(len(sur[1])):
        sur_sum = np.sum(sur[:,i])
        if sur_sum > 0:
            surprise.append(1)
        else:
            surprise.append(0)
    return surprise

def get_surprise_sum(player_list,i):
    pw = get_data(player_list,f'p{i}_winner')
    npw = get_data(player_list,f'p{i}_net_pt_won')
    bp = get_data(player_list,f'p{i}_break_pt')
    bpw = get_data(player_list,f'p{i}_break_pt_won')
    sur = [pw,npw,bp,bpw]
    sur = np.array(sur)
    surprise = []
    for i in range(len(sur[1])):
        sur_sum = np.sum(sur[:,:i])
        surprise.append(sur_sum)
    return surprise

def get_server_change(player_list,i):
    serve = get_data(player_list,'server')
    serve = [1 if x == i else 0 for x in serve]
    change = [serve[0]-1]
    for i in range(1,len(serve)):
        if serve[i] == 0:
            change.append(-1)
        else:
            if serve[i] == serve[i-1]:
                change.append(-1)
            else:
                change.append(0)
    return change