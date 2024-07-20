import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import python.data_reader as mDR

def read_and_combine_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def calculate_momentum(combined_df):
    player_list = []
    for index, row in combined_df.iterrows():
        player_data = row.to_dict()
        player_list.append(player_data)
    p1m, p2m = mDR.getMomentum(player_list)
    return p1m, p2m

def Data_process(file_path, max_shift):
    combined_df = read_and_combine_data(file_path)
    # 在最后一列加上计算好的两个玩家的momentum
    # combined_df = append_momentum(combined_df, max_shift)
    # 准备训练数据 -> 当前的特征+之前五个的momentum
    features = combined_df[[
                            # 'p1_points_won', 'p2_points_won', \
                            # 'p1_double_fault', 'p2_double_fault', \
                            # 'p1_distance_run', 'p2_distance_run', \
                            # 'p1_break_pt_won', 'p2_break_pt_won', \
                            # 'p1_break_pt_missed', 'p2_break_pt_missed', \
                            # 'p1_unf_err','p2_unf_err',\
                            # 'p1_sets', 'p2_sets', \
                            # 'p1_games', 'p2_games', \
                            # 'p1_score', 'p2_score', \
                            #
                            # 'p1_winner', 'p1_ace', \
                            # 'Player1_Momentum', \
                            'Player1_Momentum_shift_1', \
                            'Player1_Momentum_shift_2', \
                            'Player1_Momentum_shift_3', \
                            'Player1_Momentum_shift_4']]
    labels = combined_df['Player1_Momentum']
    return features, labels

def find_turning_points(p1m, p2m, window_size=5):
    length = len(p1m)
    is_p1 = p1m[1] > p2m[1]
    turning_indexes=[]
    turning_player=[]
    for i in range(length - window_size):
        if is_p1 and p1m[i] < p2m[i]:
            tmp_bool = False
            for j in range(i, i+window_size):
                if p1m[j] > p2m[j]:
                    i = j
                    tmp_bool = True
                    break
            if not tmp_bool:
                is_p1 = False
                turning_indexes.append(i)
                turning_player.append(is_p1)
                i = j
        if (not is_p1) and p2m[i] < p1m[i]:
            tmp_bool = False
            for j in range(i, i + window_size):
                if p1m[j] < p2m[j]:
                    i = j
                    tmp_bool = True
                    break
            if not tmp_bool:
                is_p1 = True
                turning_indexes.append(i)
                turning_player.append(is_p1)
                i = j
    return turning_indexes, turning_player
    # combined_df = pd.DataFrame({'Player1': p1m, 'Player2': p2m})
    #
    # # 绘制momentum图像
    # plt.figure(figsize=(10, 6))
    # plt.plot(combined_df['Player1'], label='Player 1 Momentum')
    # plt.plot(combined_df['Player2'], label='Player 2 Momentum')
    # plt.title('Momentum Comparison')
    # plt.xlabel('Time')
    # plt.ylabel('Momentum')
    # plt.legend()
    # plt.show()
    #
    # # 计算转折点
    # turning_points = []
    # for i in range(len(combined_df) - window_size + 1):
    #     window_data = combined_df.iloc[i:i + window_size]
    #     player1_sum = window_data['Player1'].sum()
    #     player2_sum = window_data['Player2'].sum()
    #
    #     if player2_sum > player1_sum:
    #         turning_points.append(i + window_size // 2)
    #
    # # 绘制带有转折点标记的图像
    # plt.figure(figsize=(10, 6))
    # plt.plot(combined_df['Player1'], label='Player 1 Momentum')
    # plt.plot(combined_df['Player2'], label='Player 2 Momentum')
    # plt.scatter(turning_points, combined_df.iloc[turning_points]['Player2'], color='red', label='Turning Point')
    # plt.title('Momentum Comparison with Turning Points')
    # plt.xlabel('Time')
    # plt.ylabel('Momentum')
    # plt.legend()
    # plt.show()

    # return turning_points

# 示例用法
# 假设p1m和p2m是两位玩家的momentum数据
# turning_points = find_turning_points(p1m, p2m)
#
# if __name__ == '__main__':
#     player_list = mDR.getList('../statics/29splits/session1.csv')
#     p1m, p2m = mDR.getMomentum(player_list)
#
#     p1c, p2c = mDR.getP1P2SetScore(player_list)
#     turning_points, tplayer = find_turning_points(p1m, p2m, 25)
#     tlp = np.ones(len(turning_points))
#     colors = ['#1f77b4' if val else '#FF7F0E' for val in tplayer]
#     print(turning_points)
#     plt.figure()
#     plt.subplot(3,1,1)
#     plt.plot(p1m)
#     plt.plot(p2m)
#     plt.xlim([0,250])
#     plt.legend(['p1m','p2m'])
#
#     plt.subplot(3,1,2)
#     plt.scatter(turning_points, tlp,c=colors)
#
#     plt.xlim([0, 250])
#
#     plt.subplot(3,1,3)
#     plt.plot(p1c)
#     plt.plot(p2c)
#     plt.xlim([0, 250])
#     plt.legend(['p1 score','p2 score'])
#
#     plt.savefig('TurningPoints.eps')
#     plt.show()
if __name__ == '__main__':
    player_list = mDR.getList('../statics/29splits/session1.csv')
    p1m, p2m = mDR.getMomentum(player_list)

    p1c, p2c = mDR.getP1P2SetScore(player_list)
    turning_points, tplayer = find_turning_points(p1m, p2m, 25)
    tlp = np.ones(len(turning_points))
    colors = ['#1f77b4' if val else '#FF7F0E' for val in tplayer]
    print(turning_points)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(p1m)
    plt.plot(p2m)
    plt.xlim([0, 250])
    plt.legend(['p1m', 'p2m'])

    plt.subplot(3, 1, 2)
    plt.scatter(turning_points, tlp, c=colors)
    # Set the y-limits to span the height of the subplot
    plt.ylim([0, 1.5])

    # Draw rectangles with adjusted ymin and ymax
    for i in range(len(turning_points) - 1):
        plt.axvspan(xmin=turning_points[i], xmax=turning_points[i + 1], ymin=0, ymax=1, facecolor=colors[i], alpha=0.5)
    plt.xlim([0, 250])

    plt.subplot(3, 1, 3)
    plt.plot(p1c)
    plt.plot(p2c)
    plt.xlim([0, 250])
    plt.legend(['p1 score', 'p2 score'])

    plt.savefig('TurningPoints.eps')
    plt.show()
