# 靡不有初，鲜克有终
import copy
import random
import numpy as np
import pandas as pd
import sys
np.seterr(invalid='ignore')

# 输入节点id(列表：[1,2,...])和时间成本矩阵，输出访问序列和访问时间

# 显示求解进度
def progress_bar(process_rate):
    process = int(100*process_rate//1+1)
    print("\r", end="")
    print("蚁群算法solveTSP当前计算进度: {}%: ".format(process), "▋" * (process // 2), end="")
    sys.stdout.flush()


# 按照轮盘赌算法随机选择下一个目标:输入各个概率组成的列表，输出所选择对象的索引
def random_selection(rate):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    rate_new = copy.deepcopy(rate)
    for i in range(0, len(rate_new)):
        rate_new[i] = (10**5)*round(rate[i], 5)  # 将概率扩大一定倍数，以便符合仿真的数据结构
    start = 0
    index = 0
    randnum = random.randint(1, int(sum(rate_new)))
    for index, scope in enumerate(rate_new):
        start += scope
        if randnum <= start:
            break
    return index


# 计算每一个OD结点对的选择概率:输入弗洛蒙浓度矩阵tau_m(tau_mtx)、能见度矩阵eta_mtx(eta_mtx)、搜索禁忌列表taboo_l(taboo_lst)、当前结点索引，输出选择概率列表
def calculate_probability(tau_m, eta_m, taboo_l, current_index):
    alpha, beta = 1, 1  # 信息素指数,可见度指数
    numerator_lst = [0 for i in range(0, len(tau_m))]  # 初始化分子列表
    prob_lst = [0 for i in range(0, len(tau_m))]  # 初始化选择概率列表
    for i in range(0, len(taboo_l)):
        if taboo_l[i] == 1:  # 根据禁忌表中，可以访问的城市，访问过的城市为0，没有的为1，1表示可以访问
            numerator_lst[i] = (tau_m[current_index, i] ** alpha) * (eta_m[current_index, i] ** beta)+0.1**50  # 分子的计算
    total = sum(numerator_lst)
    for i in range(0, len(prob_lst)):
        prob_lst[i] = numerator_lst[i] / total
    return prob_lst


# 蚁群算法，单只蚂蚁的视角:输入起点，和初始弗洛蒙矩阵，不重复地走完所有其他结点，最后回到起点，输出此时的距离dis
def ant_colony_optimization(start_index, tau_m, eta_mtx):
    current_index = start_index
    path_index = []  # 用于存储走过的【结点】索引
    taboo_lst = [1 for i in range(0, len(tau_m))]  # len(tau_m)就是待访问的节点数量
    taboo_lst[start_index] = 0
    path_index.append(current_index)
    while sum(taboo_lst) != 0:  # 不重复地访问除了起点之外的所有结点
        next_index = random_selection(calculate_probability(tau_m, eta_mtx, taboo_lst, current_index))  # 下一个结点的索引
        taboo_lst[next_index] = 0
        path_index.append(next_index)
        current_index = next_index
    path_index.append(start_index)  # 由若干结点的索引，构成的路径
    return path_index


# 2-邻边交换算法（给定一个TSP回路，将回路中2条非邻接边，用不在回路中的2条边替代，得到一条长度更短的新TSP回路，循环直至满足终止条件）
# 用于改进蚁群算法的结果,输入每只蚂蚁的路径结果一维列表【path】，输出调整之后的这只蚂蚁的【路径策略】和【访问节点的时间戳列表】
def two_opt_algorithm(path, cost_mtx):
    opt_rounds = 20  # 邻边调整算法的参数：迭代条件20次
    time_cost = 0  # 每一只蚂蚁调整后的TSP访问时间
    time_stamp = [0]
    for k in range(0, opt_rounds):  # 假设进行opt_rounds轮改进，因为每一次切片对调换完之后，由于中间一系列节点也跟着对调，可能会影响之前的其他结果，所以应当进行多次循环
        flag = 0  # 2-邻边算法的退出标志
        for i in range(1, len(path) - 4):  # len(path)=102 ; city_number = 101; len(dis_mtx)=101
            for j in range(i + 2, len(path) - 2):
                if (cost_mtx[path[i], path[j]] + cost_mtx[path[i + 1], path[j + 1]]) < (cost_mtx[path[i], path[i + 1]] + cost_mtx[path[j], path[j + 1]]):
                    path[i + 1:j + 1] = path[j:i:-1]  # [i+1:j+1]包括了[i+1,i+2,...,j];[j:i:-1]包括了[j,j-1,...,i+1],切片的左闭右开特性
                    flag = 1
        if flag == 0:
            break
    for i in range(0, len(path) - 1):
        time_cost += cost_mtx[path[i], path[i + 1]]
        time_stamp.append(time_cost)
    return path, time_stamp


# 蚁群算法，输入【待访问节点id序列】，【时间矩阵（成本矩阵）】
# 输出蚁群算法求解之后的【访问节点id序列】、【访问时间（累计成本）】
def ACOforTSP(node_list, cost_mtx):
    # 蚁群算法参数设置
    rounds, ant_number = 100, 30  # 蚁群算法轮数，蚂蚁数
    rho, Q = 0.85, 1  # 消散系数、信息素更新常量
    node_number = len(node_list)  # 待访问节点的数量  # 1个车场、c个客户点

    tau_mtx = np.ones((node_number, node_number))  # 信息素
    eta_mtx = np.zeros((node_number, node_number))  # 能见度

    shortest_dis_lst = []  # 单轮次所有蚂蚁的最短距离
    optimal_policy = []  # 初始化全局次内，蚂蚁的最优行驶路径
    optimal_time_stamp = []  # 初始化全局次内，蚂蚁的最优策略对应的访问时间戳
    new_optimal_policy = []
    new_optimal_time_stamp = []
    times = 1  # 仿真轮次初始值

    # 进行迭代
    while times <= rounds:
        policy_mtx = []  # 初始化每只蚂蚁的行驶路径，里面具有蚂蚁数量个的路径（每一个都是由结点序列组成的列表）
        time_stamp_mtx = []
        sigle_round_dis_lst = []  # 单轮次中，记录每只蚂蚁的访问总距离
        progress_bar(times / rounds)
        tau_mtx_round = copy.deepcopy(tau_mtx)  # 在同一轮的概率计算中所使用的不变的弗洛蒙矩阵
        tau_mtx = copy.deepcopy(tau_mtx * rho)
        # 每一只蚂蚁进行优化求解其初始路线
        for i in range(0, ant_number):
            start_index = random.randint(0, node_number - 1)  # 随机生成蚂蚁的起点
            policy = ant_colony_optimization(start_index, tau_mtx_round, eta_mtx)  # 单只蚂蚁ACO的初始解
            policy, time_stamp = two_opt_algorithm(policy, cost_mtx)  # 对其进行2-opt调整,time_stamp是列表
            policy_mtx.append(policy)  # policy_mtx里面是在当前这一轮中，每一个蚂蚁跑出来的路径[1,17,34...]组成的二维列表[[],[],[]...]
            time_stamp_mtx.append(time_stamp)  # 时间戳同上
            sigle_round_dis_lst.append(time_stamp[-1])  # 存储了每一只蚂蚁的总行驶成本

        shortest_dis_lst.append(min(sigle_round_dis_lst))  # 用来存储这一轮中n个蚂蚁的最短距离
        shortest_till_now = min(shortest_dis_lst)  # 截止到目前的最短距离
        optimal_policy_index = sigle_round_dis_lst.index(min(sigle_round_dis_lst))  # 找到当前轮次最短行驶距离对应的蚂蚁索引
        optimal_policy_round = policy_mtx[optimal_policy_index]  # 当前轮次的最优策略
        optimal_time_stamp_round = time_stamp_mtx[optimal_policy_index]  # 当前轮次最优策略对应的时间戳

        if min(sigle_round_dis_lst) == shortest_till_now:
            optimal_policy = optimal_policy_round  # 此时该轮最优策略即为optimal_policy
            optimal_time_stamp = optimal_time_stamp_round

        times += 1

        # 为了统一，将最终的结果调整为起点0，终点c+1的列表
        if optimal_policy[0] == 0:
            optimal_policy[-1] = node_number
            new_optimal_policy = optimal_policy
        else:
            lst1 = optimal_policy[optimal_policy.index(0):]  # 例如 [8, 7, 0, 5, 3, 1, 2, 4, 6, 8]则 [0, 5, 3, 1, 2, 4, 6, 8]
            lst2 = optimal_policy[1:optimal_policy.index(0)]  # 0的左侧且剔除了第一个[7]
            new_optimal_policy = lst1 + lst2  # [0, 5, 3, 1, 2, 4, 6, 8, 7]
            new_optimal_policy.append(node_number)

            optimal_time_stamp1 = optimal_time_stamp[optimal_policy.index(0):]
            optimal_time_stamp2 = optimal_time_stamp[1:optimal_policy.index(0)]
            for i in range(0, len(optimal_time_stamp1)):
                optimal_time_stamp1[i] = optimal_time_stamp1[i] - optimal_time_stamp[optimal_policy.index(0)]
            for i in range(0, len(optimal_time_stamp2)):
                optimal_time_stamp2[i] = optimal_time_stamp2[i] + optimal_time_stamp1[-1]
            new_optimal_time_stamp = optimal_time_stamp1 + optimal_time_stamp2
            new_optimal_time_stamp.append(new_optimal_time_stamp[-1]+cost_mtx[new_optimal_policy[-2]][node_number])

    return new_optimal_policy, new_optimal_time_stamp


if __name__ == "__main__":

    c = 3
    data_df = pd.read_csv(r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\54.问题场景挖掘及综述（二）：FSTSP&PDSTSP建模及其求解\程序代码\data\C101network.txt')
    # print(data_df)
    node_list = [i for i in range(0, c+1)]
    # print(node_list)
    # 填补成本矩阵
    costMatrix = np.zeros((c+2, c+2))
    for i in range(0, c+2):
        for j in range(0, c+2):
            if i != j:
                i_x,i_y= data_df.loc[i,'XCOORD'],data_df.loc[i, 'YCOORD']
                j_x,j_y = data_df.loc[j,'XCOORD'],data_df.loc[j, 'YCOORD']
                costMatrix[i][j] = ((i_x - j_x) ** 2 + (i_x - j_x) ** 2) ** 0.5
            else:
                pass
    print(costMatrix)
    optimal_policy, optimal_time_stamp = ACOforTSP(node_list, costMatrix)
    print()
    print("路径：",optimal_policy)
    print("对应的时间戳：",optimal_time_stamp)
