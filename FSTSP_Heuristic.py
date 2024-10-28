# 靡不有初，鲜克有终
''' FSTSP,模型服从以下假设：
a.单车场
b.单辆卡车
c.单架无人机
d.无车容量
e.无时间窗
f.无服务时间
g.无需求量
数据为所罗门C101，无人机可访问的点是随机生成并保存的
使用2-邻边改进的蚁群算法来对TSP进行求解'''
import numpy as np
import copy
import pandas as pd
from solveTSP import ACO
import matplotlib.pyplot as plt


'''定义数据类。建立用于存放和调用输入data的数据结构框架'''
class Data:
    def __init__(self):  # 建立类属性
        self.nodeNum = 0  # 节点总数量（客户点、起点车场、终点车场）
        self.N_lst = []  # 存放所有节点的id的列表,N={0,1,2,...,c+1}

        self.corX = []  # 节点横坐标
        self.corY = []  # 节点纵坐标
        self.costMatrix = None  # 节点与结点之间的距离矩阵

        self.corX_UAV = []  # 适合UAV的客户点横坐标
        self.corY_UAV = []  # 适合UAV的客户点横坐标

        self.corX_Truck = []  # 不适合UAV的客户点横坐标
        self.corY_Truck = []  # 不适合UAV的客户点横坐标

        self.N_outNum = 0  # 可流出节点的总数量（除了终点车场）
        self.N_out_lst = []  # 存放所有可流出节点的id的列表,N0={0,1,2,...,c}

        self.N_inNum = 0  # 可流入节点的总数量（除了起点车场）
        self.N_in_lst = []  # 存放所有可流入节点的id的列表,N+={1,2,...,c+1}

        self.customerNum = 0  # 客户总数量 = c
        self.C_lst = []  # 存放所有客户点的id的列表,C={1,2,...,c}

        self.UAVCustomerNum = 0  # 可被无人机访问的客户总数量
        self.C_UAV_lst = []  # 存放所有可被无人机访问的客户id的列表 C'∈ C

        self.sortiesNum = 0  # 无人机可执行架次的数量
        self.P_lst = []  # 无人机可执行架次的集合


'''读取数据。将对应的输入数据存放至对应的数据结构框架中。函数参数包括文件路径、客户点数量（客户点在1-100之间）、无人机续航里程'''
def readData(file_path, customerNum, UAV_endurance):
    data = Data()
    data.customerNum = customerNum
    data_df = pd.read_csv(file_path)
    # 将1个起始车场（文本数据中的第一个）+ customerNum 个客户点的信息存放在对应数据结构中,此外也会建立一个虚拟终点车场
    for i in range(0, data.customerNum+1):
        data.N_lst.append(data_df.loc[i, 'CUST NO']-1)  # 从0开始的所有节点，N={0,1,...,c+1}
        data.corX.append(data_df.loc[i, 'XCOORD'])
        data.corY.append(data_df.loc[i, 'YCOORD'])
        if data_df.loc[i, 'DRONE'] == 1:
            data.C_UAV_lst.append(data_df.loc[i, 'CUST NO']-1)  # 存储所有可以被无人机访问的客户点
            data.corX_UAV.append(data_df.loc[i, 'XCOORD'])
            data.corY_UAV.append(data_df.loc[i, 'YCOORD'])
        else:
            data.corX_Truck.append(data_df.loc[i, 'XCOORD'])
            data.corY_Truck.append(data_df.loc[i, 'YCOORD'])

    # 增加虚拟终点车场并建立其他集合
    data.N_lst.append(customerNum+1)  # 从0开始的所有节点，N={0,1,...c}+{c+1}
    data.C_lst = copy.deepcopy(data.N_lst[1:-1])
    data.corX.append(data_df.loc[0, 'XCOORD'])
    data.corY.append(data_df.loc[0, 'YCOORD'])

    data.nodeNum = len(data.N_lst)  # 客户点数量+2：N={0,1,...,c+1}
    data.UAVCustomerNum = len(data.C_UAV_lst)  # 无人机可以访问的客户点数量
    data.N_out_lst = copy.deepcopy(data.N_lst[0:data.nodeNum-1])  # N0={0,1,...,c}
    data.N_outNum = len(data.N_out_lst)
    data.N_in_lst = copy.deepcopy(data.N_lst[1:data.nodeNum])  # N+={1,...,c+1}
    data.N_inNum = len(data.N_in_lst)

    # 填补成本矩阵
    data.costMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            if i != j:
                data.costMatrix[i][j] = ((data.corX[i]-data.corX[j])**2+(data.corY[i]-data.corY[j])**2)**0.5
            else:
                pass

    # 建立无人机可行架次集合[(i,j,k),(i,j,k)....]
    for i in data.N_out_lst:
        for j in data.C_UAV_lst:
            if i != j:
                for k in data.N_in_lst:
                    if k != j and k != i and data.costMatrix[i][j] + data.costMatrix[j][k] < UAV_endurance:
                        data.P_lst.append((i, j, k))
    data.sortiesNum = len(data.P_lst)  # 无人机可访问的架次总数
    print("costMatrix:")
    print(data.costMatrix)
    return data

# 函数0：计算从卡车路径中移除j，卡车到达某点k的时间
# 输入：卡车路径、卡车到达每一个时间节点的时间戳t、节点j的id、节点k的id
# 输出：在卡车路径中移除j，卡车到达某点k的时间戳
def calcTimeStamp(j,t,truckRoute,k):
    j_index = truckRoute.index(j)
    k_index = truckRoute.index(k)
    if k_index < j_index:  # k在j前面，删除j之后k的时间戳不受到影响
        t_k_ = t[k_index]
        return t_k_
    if k_index > j_index:  # k在j后面，删除j之后k的时间戳受到影响
        j_pre = truckRoute[j_index - 1]  # 前驱节点的id
        j_suc = truckRoute[j_index + 1]  # 后继节点的id
        savings = data.costMatrix[j_pre][j] + data.costMatrix[j][j_suc] - data.costMatrix[j_pre][j_suc]
        t_k_ = t[k_index] - savings  # 计算当j从卡车路径中剔除，卡车到达节点k的时间
        return t_k_


# 函数1：计算从卡车路径中移除部分客户j所实现的增益saving
# 输入：j——当前分配给卡车的节点（待移除节点,j属于C‘）、truckRoute——卡车路径序列(列表)、t——卡车到达每一个节点的时间戳（列表）、truckSubRoutes卡车子路径集合
# 输出：在卡车路径中移除节点j对应的增益saving,原先路径中j节点的前驱节点，j节点的后继节点
def calcSaving(j, t, truckRoute, truckSubRoutes,truckSubRoutes_type):
    j_index = truckRoute.index(j)
    j_pre = truckRoute[j_index-1]  # 前驱节点的id
    j_suc = truckRoute[j_index+1]  # 后继节点的id
    savings = data.costMatrix[j_pre][j] + data.costMatrix[j][j_suc] - data.costMatrix[j_pre][j_suc]  # 定义将节点j从卡车路径中去除的增益（现在的成本-去除之后的成本）
    for i in range(0, len(truckSubRoutes)):
        if len(truckSubRoutes_type[i]) > 1 and j in truckSubRoutes[i]: # 节点j位于无人机架次对应的卡车子环路中
            a = truckSubRoutes_type[i][0]  # 节点j所位于的，该卡车子路径中起点id（飞机发射点id）定义为a
            b = truckSubRoutes_type[i][2]  # 节点j所位于的，该卡车子路径中终点id(飞机回收点id)定义为b
            j_UAV = truckSubRoutes_type[i][1]  # 节点j所位于的，该卡车子路径在的中间点id(飞机访问客户点id)定义为j_UAV
            # 确定最终删除节点j，使卡车获得的时间增益，木桶短板取下界（因为可能有卡车等待无人机）
            a_index = truckRoute.index(a)
            t_b_ = calcTimeStamp(j,t,truckRoute,b) # 计算当j从卡车路径中剔除，卡车到达节点b的时间
            t_a = t[a_index]
            savings = min(savings, t_b_-(t_a+UAV_factor*data.costMatrix[a][j_UAV]+UAV_factor*data.costMatrix[j_UAV][b]+SR))
        else:  # 节点j不位于无人机架次对应的卡车子环路中
            pass
    return savings,j_pre,j_suc


# 函数2：在子路径subroute是与UAV联合的情况下，选择使用卡车对节点j提供服务，计算节点j在subroute中的最佳插入位置，及其对应的插入成本cost
# 输入：在上一步被移除的节点j，卡车到达每一个节点的时间戳t，待回插的车辆子路径
# 输出：将节点j插入待回插子路径的最大增益，以及待回插位置，j被服务的情况（是否被无人机服务）
def calcCostTruck(j,t,subroute,savings,j_insert_lst, maxSavings,truckRoute):
    servedByUAV = False  # 只能用卡车，不能用UAV
    [i_optimal, j_optimal, k_optimal] = j_insert_lst
    a = subroute[0]  # 将子路径的起点id定义为a
    b = subroute[-1]  # 将子路径的终点id定义为b
    a_index = truckRoute.index(a)
    b_index = truckRoute.index(b)
    for i in range(0,len(subroute)-1):  # 对于子路径中的所有两个相邻的点对（i，k）
        i_id = subroute[i]
        k = i+1
        k_id = subroute[k]
        # 将j插入（i，k）之间，则成本：点j的插入使子路径增加的时间（插入之后花费的时间-现在花费的时间）,如果i和k本身就是j的前节点和后继，那就跳过
        cost = data.costMatrix[i_id][j] + data.costMatrix[j][k_id] - data.costMatrix[i_id][k_id]
        if cost < savings and i_id !=j and k_id !=j and i_id != i_optimal and k_id != k_optimal:  # 这里的cost是插入后-插入前，saving是删除前-删除后，方向不同
            # 说明将j删除的增益，是大于将j插入此处的成本，即有潜在净增益，可以插入
            # 但要检验j的插入对无人机飞行时间的影响
            if t[a_index] - t[b_index] + cost <= UAV_endurance:  # 没有超过无人机续航时间
                if savings - cost > maxSavings:
                    servedByUAV = False  # 说明j没有被无人机服务，而是被插入进了卡车的子路径中，由卡车提供服务
                    i_optimal = i_id  # i*，j的前驱节点
                    j_optimal = j  # j*
                    k_optimal = k_id  # k*，j的后继节点
                    maxSavings = savings - cost  # 净增益
                else:
                    pass
            else:
                pass
        else:
            pass

    return [maxSavings,i_optimal,j_optimal,k_optimal,servedByUAV]


# 函数3：在子路径subroute不是与UAV联合的情况下，选择使用UAV对节点j提供服务，计算节点j在subroute中的最佳架次，及其对应的插入成本cost
# 输入：在上一步被移除的节点j，卡车到达每一个节点的时间戳t，待回插的车辆子路径subroute，以及目前的savings,卡车原有的行驶路径truckRoute
# 输出：将节点j添加到子路径中对应无人机架次的最大增益，以及待架次位置，j被服务的情况（是否被无人机服务）
def calcCostUAV(j,t,subroute,savings, truckRoute, j_insert_lst,maxSavings):
    servedByUAV = False  # 初始化
    [i_optimal, j_optimal, k_optimal] = j_insert_lst
    for i in range(0,len(subroute)-1):
        i_id = subroute[i]
        i_index = truckRoute.index(i_id)
        for k in range(i+1,len(subroute)):  # 对于子路径中两个点i和k（i在k前即可）
            k_id = subroute[k]
            if UAV_factor*(data.costMatrix[i_id][j] + data.costMatrix[j][k_id]) <= UAV_endurance and i_id!=j and k_id !=j: # 在无人机续航内，可以由无人机执行
                # 计算当移除掉j时，卡车到达节点k的时间戳
                t_k_ = calcTimeStamp(j, t, truckRoute, k_id)
                t_i_ = t[i_index]
                max1 = (t_k_ - t_i_) + SL + SR
                max2 = SL + SR + UAV_factor * (data.costMatrix[i_id][j] + data.costMatrix[j][k_id])
                max3 = max(max1, max2) - (t_k_ - t_i_)
                cost = max(0, max3)
                if savings-cost > maxSavings:
                    servedByUAV = True  # 说明j被无人机服务
                    i_optimal = i_id  # i*，j的前驱节点(UAV发射点)
                    j_optimal = j  # j*
                    k_optimal = k_id  # k*，j的后继节点（UAV回收点）
                    maxSavings = savings - cost  # 净增益
                else:
                    pass
            else:
                pass
    return [maxSavings, i_optimal, j_optimal, k_optimal, servedByUAV]


# 函数4：根据j的删除，以及安排卡车/安排UAV对应的情况，更新自圈集合
# 输入：节点j是被无人机访问还是被卡车访问的指示变量servedByUAV，最佳位置i_optimal，j_optimal， k_optimal
# 输出：删除并重新回插j节点后，子路径的集合，以及更新和调整之后的时间戳t
def performUpdate(servedByUAV,i_optimal,j_optimal,k_optimal,truckRoute,t,truckSubRoutes,Cprime,truckSubRoutes_type):
    if servedByUAV == True:
        # 更新将j从truckRoute中删除之后的t_new.
        t_new = []
        for k in range(0, len(truckRoute)):
            k_id = truckRoute[k]
            if k_id != j_optimal:
                t_new_k = calcTimeStamp(j_optimal, t, truckRoute, k_id)  # 计算从卡车路径中移除j，卡车到达某点k的时间
                t_new.append(t_new_k)

        # 在卡车原始路径truckRoute(TSP求出来的)、卡车子路径集合truckSubRoutes中，删除j
        truckRoute.remove(j_optimal)
        for i in range(0,len(truckSubRoutes)):
            if j_optimal in truckSubRoutes[i]:
                truckSubRoutes[i].remove(j_optimal)
            else:
                pass
        # 截断原来的子路径


        truckSubRoutes_new = copy.deepcopy(truckSubRoutes)
        for i in range(0,len(truckSubRoutes_new)):
            if (i_optimal in truckSubRoutes[i]) and (k_optimal in truckSubRoutes[i]) and (j_optimal not in truckSubRoutes[i]): # 如果无人机起点在路径中
                i_optimal_index = truckSubRoutes[i].index(i_optimal)
                k_optimal_index = truckSubRoutes[i].index(k_optimal)

                new_subRoute1 = truckSubRoutes[i][0 :i_optimal_index+1].copy()
                new_subRoute2 = truckSubRoutes[i][i_optimal_index :k_optimal_index+1].copy()
                new_subRoute3 = truckSubRoutes[i][k_optimal_index:len(truckSubRoutes[i])+1].copy()
                truckSubRoutes.remove(truckSubRoutes[i])
                del truckSubRoutes_type[i]

                if len(new_subRoute1) >1 :
                    truckSubRoutes.append(new_subRoute1)
                    truckSubRoutes_type.append([0])
                if len(new_subRoute2) >1 :
                    truckSubRoutes.append(new_subRoute2)
                    truckSubRoutes_type.append((i_optimal,j_optimal,k_optimal))
                if len(new_subRoute3) >1 :
                    truckSubRoutes.append(new_subRoute3)
                    truckSubRoutes_type.append([0])


        # 将无人机架次对应的卡车架次[i_optimal→k_optimal]添加到卡车子路径集合truckSubRoutes中
        #truckSubRoutes.append(new_subRoute2)
        #truckSubRoutes_type.append((i_optimal,j_optimal,k_optimal)) # 这里与truckSubRoutes中对应，[0]的话是卡车路径，（i,j,k）则是无人机路径，可用此来判断子路径类型
        # 将i_optimal,j_optimal,k_optimal从Cprime中移除（将i、j、k保护起来了，再也不会被删除了）
        Cprime.remove(j_optimal)
        if i_optimal in Cprime:
            Cprime.remove(i_optimal)
        if k_optimal in Cprime:
            Cprime.remove(k_optimal)

    else:  # 选择j通过卡车来访问
        # 将j从其所位于的卡车子路径中删除
        for i in range(0, len(truckSubRoutes)):
            if j_optimal in truckSubRoutes[i]:
                truckSubRoutes[i].remove(j_optimal)
            else:
                pass
        # 将j插入i和k之间的子路径
        for i in range(0, len(truckSubRoutes)):
            if (i_optimal in truckSubRoutes[i]) and (k_optimal in truckSubRoutes[i]):
                j_insert_index = truckSubRoutes[i].index(i_optimal)+1
                truckSubRoutes[i].insert(j_insert_index,j_optimal)
            else:
                pass

        # 更新重新将j删除并插入i和k之间的，原始truckRoute
        print("__servedByUAV", servedByUAV)
        print("__i_optimal", i_optimal)
        print("__j_optimal:",j_optimal)
        print("__k_optimal", k_optimal)
        print("__Cprime", Cprime)
        print("__t", t)
        print("__truckRoute:", truckRoute)
        print('__truckRoute:',truckSubRoutes)
        print("__truckSubRoutes_type", truckSubRoutes_type)

        truckRoute.remove(j_optimal)
        j_insert_index = truckRoute.index(i_optimal) + 1
        truckRoute.insert(j_insert_index,j_optimal)

        # 更新重新将j插入i和k之间的，t_new
        t_new = [0]
        cost = 0
        for i in range(0, len(truckRoute)-1):
            start_id = truckRoute[i]
            end_id = truckRoute[i]+1
            cost = cost + data.costMatrix[start_id][end_id]
            t_new.append(cost)
        # Cprime.remove(j_optimal)

    # 根据无人机和卡车木桶短板，考虑SL和SR，来最后矫正t_new
    for i in range(0,len(truckSubRoutes_type)):
        if truckSubRoutes_type[i] != [0]: # 是UAV的架次
            UAV_start_id = truckSubRoutes_type[i][0]
            UAV_serve_id = truckSubRoutes_type[i][1]
            UAV_end_id = truckSubRoutes_type[i][2]
            UAV_fling_durance = UAV_factor*(data.costMatrix[UAV_start_id][UAV_serve_id]+data.costMatrix[UAV_serve_id][UAV_end_id])  # UAV执行任务飞行时长
            t_start_stamp_UAV = t_new[truckRoute.index(UAV_start_id)] + SL  # 无人机起飞时间戳/也是卡车出发时间戳
            t_arive_stamp_UAV = t_start_stamp_UAV + UAV_fling_durance + SR  # 无人机到达时间戳/应当与卡车到达时间戳进行比较

            t_new[truckRoute.index(UAV_start_id)] = t_start_stamp_UAV  # 矫正无人机发射点的卡车时间戳
            for j in range(truckRoute.index(UAV_start_id)+1,len(t_new)):  # 其后面的时间戳均增加SL
                t_new[j] = t_new[j]+SL
            t_arive_stamp_truck = t_new[truckRoute.index(UAV_end_id)]  #卡车到达回收点时间戳
            if t_arive_stamp_UAV > t_arive_stamp_truck: # 在回收点无人机时间戳大，说明无人机到得晚，此时卡车需要等待无人机，因此卡车时间戳t_new会继续调整
                delay = t_arive_stamp_UAV - t_arive_stamp_truck  # 卡车需要延后的时间
                for k in range(truckRoute.index(UAV_end_id), len(t_new)):  # 其后面的时间戳均增加SL
                    t_new[k] = t_new[k] + delay
            else:
                pass
    return t_new,truckRoute,truckSubRoutes,Cprime,truckSubRoutes_type


def plot_show(optimal_policy_ACO, optimal_time_stamp, truckSubRoutes, t, truckSubRoutes_type):  # 绘图,输入蚁群算法得到的策略、大邻域启发式策略，输出图片
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    # ____________________对于图1____________________
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f"ACO-solution,time cost: {optimal_time_stamp[-1]}")
    ax1.scatter(data.corX[0], data.corY[0], c='black', alpha=1, marker=',', linewidths=2, label='depot')
    ax1.scatter(data.corX_Truck[1:], data.corY_Truck[1:], c='blue', alpha=1, marker='o', linewidths=1, label='customer_Truck')  # 不适合无人机的客户点
    ax1.scatter(data.corX_UAV, data.corY_UAV, c='red', alpha=1, marker='o', linewidths=1,label='customer_UAV')  # 适合无人机的客户点
    # 绘制节点的id
    for i in range(0, data.customerNum + 1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.N_lst[i]
        ax1.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom", ha='left', color='k')
    # 绘制卡车路径
    for i in range(0, len(optimal_policy_ACO) - 1):  # a→b，以索引定位
        a = optimal_policy_ACO[i]
        b = optimal_policy_ACO[i + 1]
        x = [data.corX[a], data.corX[b]]
        y = [data.corY[a], data.corY[b]]
        ax1.plot(x, y, color='blue', linewidth=1, linestyle='-')

    # ____________________对于图1____________________
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f"ACO,time cost: {optimal_time_stamp[-1]}")
    ax2.scatter(data.corX[0], data.corY[0], c='black', alpha=1, marker=',', linewidths=2, label='depot')
    ax2.scatter(data.corX_Truck[1:], data.corY_Truck[1:], c='blue', alpha=1, marker='o', linewidths=1,label='customer_Truck')  # 不适合无人机的客户点
    ax2.scatter(data.corX_UAV, data.corY_UAV, c='red', alpha=1, marker='o', linewidths=1,label='customer_UAV')  # 适合无人机的客户点
    # 绘制节点的id
    for i in range(0, data.customerNum + 1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.N_lst[i]
        ax1.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom", ha='left', color='k')
    # 绘制卡车路径
    for i in range(0, len(optimal_policy_ACO) - 1):  # a→b，以索引定位
        a = optimal_policy_ACO[i]
        b = optimal_policy_ACO[i + 1]
        x = [data.corX[a], data.corX[b]]
        y = [data.corY[a], data.corY[b]]
        ax1.plot(x, y, color='blue', linewidth=1, linestyle='-')
    ax1.legend(loc='best')

    # ____________________对于图2____________________
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f"FSTSP_Heuristic,time cost: {t[-1]}")
    ax2.scatter(data.corX[0], data.corY[0], c='black', alpha=1, marker=',', linewidths=2)
    ax2.scatter(data.corX_Truck[1:], data.corY_Truck[1:], c='blue', alpha=1, marker='o', linewidths=1)  # 不适合无人机的客户点
    ax2.scatter(data.corX_UAV, data.corY_UAV, c='red', alpha=1, marker='o', linewidths=1)  # 适合无人机的客户点
    # 绘制节点的id
    for i in range(0, data.customerNum + 1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.N_lst[i]
        ax2.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom",ha='left', color='k')
    # 绘制卡车路径
    for i in range(0, len(truckSubRoutes)):  # a→b，以索引定位
        for j in range(0, len(truckSubRoutes[i]) - 1):
            a = truckSubRoutes[i][j]
            b = truckSubRoutes[i][j+1]
            x = [data.corX[a], data.corX[b]]
            y = [data.corY[a], data.corY[b]]
            ax2.plot(x, y, color='blue', linewidth=1, linestyle='-')
        if len(truckSubRoutes_type[i]) == 3:    # 绘制无人机路径
            for k in range(0, len(truckSubRoutes_type[i]) - 1):
                a = truckSubRoutes_type[i][k]
                b = truckSubRoutes_type[i][k + 1]
                x = [data.corX[a], data.corX[b]]
                y = [data.corY[a], data.corY[b]]
                ax2.plot(x, y, color='red', linewidth=1, linestyle=':')
    ax2.legend(loc='best')

    plt.grid(False)
    plt.show()


def FSTSP_Heuristic():  # 主函数
    Cprime = data.C_UAV_lst
    truckRoute, t = ACO.ACOforTSP(data.N_out_lst, data.costMatrix)
    ACO_route = copy.deepcopy(truckRoute)
    ACO_t = copy.deepcopy(t)
    truckSubRoutes = [truckRoute]  # 卡车访问的子路径集合 如 [[8, 7, 0],[0, 5, 3, 1, 2, 4, 6, 9]]
    truckSubRoutes_type = [[0]]  # 这里与truckSubRoutes中对应，[0]的话是卡车路径，（i,j,k）则是无人机路径，可用此来判断子路径类型
    maxSavings = 0
    flag = 0  # 循环终止条件
    servedByUAV = False
    while flag != 1:
        for j in Cprime:  # 对于每一个可以由无人机去访问的节点（j是id）,首先计算当移除它的时候，增益saving是多少
            savings,j_pre,j_suc = calcSaving(j, t, truckRoute, truckSubRoutes, truckSubRoutes_type)
            i_optimal,j_optimal,k_optimal=j_pre,j,j_suc
            for subroute in truckSubRoutes:  # 对于每一个子路径集合中的子路径，如[[8, 7, 0],[0, 5, 3, 1, 2, 4, 6, 9]]中的[8, 7, 0]
                subroute_index = truckSubRoutes.index(subroute)
                if len(truckSubRoutes_type[subroute_index]) > 1: # 如果此子路径是与UAV联合的，那么只能用卡车
                    # 初始化，用的是j原来在卡车路径中的情况，以防止在净增益不大于0时，servedByUAV和j_insert_lst没有输出
                    j_insert_lst = [j_pre, j, j_suc]  #  j节点在原有路径中的顺序
                    [a,b,c,d,e] = calcCostTruck(j, t, subroute, savings, j_insert_lst,maxSavings,truckRoute)
                    if a > maxSavings:  # 此时说明j节点在最佳插入情况下，净增益大于0，那么就需要存储并覆盖更新现有较好的j的插入决策
                        maxSavings = a  # 覆盖更新maxSavings
                        i_optimal = b # 覆盖更新i_optimal
                        j_optimal = c # 覆盖更新j_optimal
                        k_optimal = d # 覆盖更新k_optimal
                        servedByUAV = e # 覆盖更新servedByUAV，更新为由卡车去访问
                        j_insert_lst_opt = [i_optimal,j_optimal,k_optimal]  # 覆盖更新j_insert_lst
                elif len(truckSubRoutes_type[subroute_index]) == 1:  # 此子路径不是与UAV联合的，那么优先用UAV
                    # 初始化，用的是j原来在卡车路径中的情况，以防止在净增益不大于0时，servedByUAV和j_insert_lst没有输出
                    j_insert_lst = [j_pre, j, j_suc]  # j节点在原有路径中的顺序
                    [a, b, c, d, e] = calcCostUAV(j,t,subroute,savings,truckRoute,j_insert_lst,maxSavings)
                    if a > maxSavings:  # 此时说明j节点在最佳飞行访问架次情况下，净增益大于0，那么就需要存储并覆盖更新现有较好的j的插入决策
                        maxSavings = a # 覆盖更新maxSavings
                        i_optimal = b # 覆盖更新i_optimal
                        j_optimal = c # 覆盖更新j_optimal
                        k_optimal = d # 覆盖更新k_optimal
                        servedByUAV = e # 覆盖更新servedByUAV，更新为由无人机去访问
                        j_insert_lst_opt = [i_optimal,j_optimal,k_optimal]  # 覆盖更新j_insert_lst
        if maxSavings > 0:
            maxSavings = 0
            t,truckRoute,truckSubRoutes,Cprime,truckSubRoutes_type = performUpdate(servedByUAV, i_optimal, j_optimal, k_optimal, truckRoute, t, truckSubRoutes, Cprime, truckSubRoutes_type)

        elif maxSavings <= 0 or len(Cprime) == 0:
            flag = 1
    return truckRoute,truckSubRoutes,truckSubRoutes_type,t,ACO_route,ACO_t


'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\55.问题场景挖掘及综述（二）：FSTSP&PDSTSP建模及其求解\程序代码\data\C101network.txt'  # 这里是节点文件
    customerNum = 25 # C的数量
    UAV_endurance = 20  # 无人机续航
    M = 100  # 约束中存在的M,注意不能取的太大，要保证准确的基础上尽量紧凑
    SR = 1  # UAV回收时间
    SL = 1  # UAV发射时间
    UAV_factor = 0.5  # 认为无人机的速度是卡车速度的2倍，所以所花的时间是无人机的1/2
    # 读取数据
    print("-" * 20, "Problem Information", '-' * 20)
    data = readData(data_path, customerNum, UAV_endurance)
    print("data.N_lst(N):", data.N_lst)
    print("data.N_out_lst(N0):", data.N_out_lst)
    print("data.N_in_lst(N+):", data.N_in_lst)
    print("data.C_lst(C):", data.C_lst)
    print("data.C_UAV_lst(C'):", data.C_UAV_lst)
    print("data.P_lst(P):", data.P_lst)
    # 输出相关数据
    print(f'节点总数: {data.nodeNum}')
    print(f'客户点总数: {data.customerNum}')
    print(f'可被无人机服务总数: {data.UAVCustomerNum}')
    # 求解
    truckRoute,truckSubRoutes,truckSubRoutes_type,t,ACO_route, ACO_t = FSTSP_Heuristic()
    print()
    print("卡车行驶路径truckRoute:", truckRoute)
    print("卡车行驶子路径集合truckSubRoutes:", truckSubRoutes)
    print("卡车行驶子路径类型truckSubRoutes_type:", truckSubRoutes_type)
    print('此时的时间戳序列t：', t)
    print()

    plot_show(ACO_route, ACO_t,truckSubRoutes,t, truckSubRoutes_type)

