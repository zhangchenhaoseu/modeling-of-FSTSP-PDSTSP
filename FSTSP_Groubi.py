# 靡不有初，鲜克有终
''' FSTSP,模型服从以下假设：
a.单车场
b.单辆卡车
c.单架无人机
d.无车容量
e.无时间窗
f.无服务时间
g.无需求量
数据为所罗门C101，无人机可访问的点是随机生成并保存的 '''
import numpy as np
import copy
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
import time
import re


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
    print("costMatrix[4][1]:")
    print(data.costMatrix[4][1])
    return data


'''定义解类。建立求解结果的数据结构框架，并建立将Gurobi求解结果存放在数据结构中的连接'''
class Solution:
    ObjVal = 0  # 目标函数值
    X = None  # 用于存放卡车的决策变量X_ij的值
    X_routes = None  # 用于存放卡车所经过的节点序列
    Y = None  # 用于存放无人机的决策变量Y_ijk的值
    Y_routes = None  # 用于存放无人机所经过的节点序列
    X_routesNum = 0  # 卡车路径数量
    Y_routesNum = 0  # 无人机架次数量

    def __init__(self, data, model):  # 建立类属性，解类的建立需要输入数据和模型
        self.ObjVal = model.ObjVal
        self.X = [[0 for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)]  # 由于仅作存储作用方便调用，因此此处全连接即可
        self.X_routes = []
        self.Y = [[[0 for k in range(0, data.nodeNum)] for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)]
        self.Y_routes = []
        self.U = [0 for k in range(0, data.nodeNum)]
        self.T = [0 for k in range(0, data.nodeNum)]
        self.TUAV = [0 for k in range(0, data.nodeNum)]


def getSolution(data, model):  # 定义函数，从Gurobi输出的模型中获得目标函数值、决策变量X和Y、继而分别得到卡车和无人机路由的节点序列
    solution = Solution(data, model)
    # 在三维矩阵结构中存储自变量的取值，拆字段，检验是否是x，然后保存
    var_lst = model.getVars()
    print('var_lst:')
    for i in var_lst:
        if i.x !=0:
            print(i)
    for v in model.getVars():
        split_arr = re.split(r"_", v.VarName)  # 将gurobi形式的变量进行拆解，便于利用数据结构实现实现存储
        if split_arr[0] == 'X' and round(v.x) != 0:
            # print(v)
            solution.X[int(split_arr[1])][int(split_arr[2])] = v.x  # X_ij
        elif split_arr[0] == 'Y' and round(v.x) != 0:
            solution.Y[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = v.x  # Y_ijk
        elif split_arr[0] == 'U' and v.x != 0:
            solution.U[int(split_arr[1])] = v.x  # U_i
        elif split_arr[0] == 'T' and v.x != 0:
            solution.T[int(split_arr[1])] = v.x  # T_i
        elif split_arr[0] == 'TUAV' and v.x != 0:
            solution.TUAV[int(split_arr[1])] = v.x  # TUAV_i
        else:
            pass
            # print(v)
    print("卡车路由变量 solution.X:", solution.X)
    print("无人机路由变量 solution.Y:", solution.Y)
    print("卡车访问辅助变量 solution.U:", solution.U)
    print("卡车访问时间戳 solution.T:", solution.T)
    print("无人机访问时间戳 solution.TUAV:", solution.TUAV)

    # 根据求解结果存储卡车的路由序列（一维，因为只有一辆车）
    i = 0  # 车辆的位置：起点车场
    solution.X_routes = [i]  # 卡车的起点从i=0开始
    while i != data.customerNum+1:  # 当车辆的位置不在终点车场时
        for j in range(0,len(solution.X[i])):  # 含虚拟结点，N={0,1,...,c+1}，data.nodeNum = c+2
            if round(solution.X[i][j]) != 0:  # 若从节点i到j的变量不为0，说明卡车在i选择前往j，则需要将j节点的id计入路由序列
                solution.X_routes.append(j)
                i = j
    solution.X_routes[-1] = 0  # 修正，将倒数第一个虚拟终点车场的id改为0，仅仅为了便于理解而已
    solution.X_routesNum += 1
    print('卡车行驶的行驶路线 solution.X_routes:', solution.X_routes)

    # 根据求解结果存储无人机的路由序列(二维，因为有若干架次)
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            for k in range(0, data.nodeNum):
                if round(solution.Y[i][j][k]) != 0:  # 取最接近的整数，因为有的量虽然非零，但很小
                    solution.Y_routes.append((i, j, k))
                    solution.Y_routesNum += 1  # 无人机架次数
    print('无人机的飞行路线 solution.Y_routes:', solution.Y_routes)

    return solution


'''绘图。以图像展示VRP求解结果'''
def plotSolution(data, solution):
    # 绘制画布
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{data.UAVCustomerNum} UAV-eligible Customers in total {data.customerNum} Customers,single truck")
    # 绘制节点
    plt.scatter(data.corX[0], data.corY[0], c='black', alpha=1, marker=',', linewidths=2, label='depot')  # 起终点
    plt.scatter(data.corX_Truck[1:], data.corY_Truck[1:], c='blue', alpha=1, marker='o', linewidths=1, label='customer_Truck')  # 不适合无人机的客户点
    plt.scatter(data.corX_UAV, data.corY_UAV, c='red', alpha=1, marker='o', linewidths=1, label='customer_UAV')  # 适合无人机的客户点
    # 绘制节点的id
    for i in range(0,data.customerNum+1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.N_lst[i]
        plt.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom", ha='left', color='k')
    # 绘制卡车路径
    for i in range(0, len(solution.X_routes)-1):  # a→b，以索引定位
        a = solution.X_routes[i]
        b = solution.X_routes[i + 1]
        x = [data.corX[a], data.corX[b]]
        y = [data.corY[a], data.corY[b]]
        plt.plot(x, y, color='blue', linewidth=1, linestyle='-')
    # 绘制无人机路径
    for k in range(0, solution.Y_routesNum):  # 对于不同的无人机架次
        for i in range(0, 2):  # 每架次由三个点构成：(i→j→k)
            a = solution.Y_routes[k][i]
            b = solution.Y_routes[k][i + 1]
            x = [data.corX[a], data.corX[b]]
            y = [data.corY[a], data.corY[b]]
            plt.plot(x, y, color='red', linewidth=1, linestyle=':')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    return 0


'''展示计算结果。以文字展示求解结果中卡车和无人机的路径答案'''
def printSolution(data,solution):
    print('_____________________________________________')
    for index, UAV_route in enumerate(solution.Y_routes):
        cost = 0
        for i in range(len(UAV_route) - 1):
            cost += UAV_factor * data.costMatrix[UAV_route[i]][UAV_route[i + 1]]
        print(f"UAV_Route-{index + 1} : {UAV_route} , time cost: {cost}")
    print('_____________________________________________')
    cost = 0
    for i in range(0, len(solution.X_routes)-1):
        cost += data.costMatrix[solution.X_routes[i]][solution.X_routes[i + 1]]
    print(f"Truck_Route: {solution.X_routes} , time cost: {cost}")


'''建模和求解。使用Gurobi对问题进行建模'''
def modelingAndSolve(data):
    # 建立模型
    m = Model('FSTSP')

    # 模型设置：由于存在函数printSolution，因此关闭输出;以及容许误差
    m.setParam('MIPGap', 0.01)
    # m.setParam('OutputFlag', 0)

    # 定义变量：
    # Step1.建立合适的数据结构建立变量的索引，FSTSP有两个决策变量X和Y,一个MTZ辅助变量U，一个卡车时间辅助变量T，一个无人机时间辅助变量T_UAV，一个访问次序一致变量p
    '______添加决策变量X______'
    # 建立存储决策变量X的数据结构
    X = [[[] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)]  # x_ij
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_out_lst:
        for j in data.N_in_lst:
            if i != j:
                X[i][j] = m.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}")
    '______添加决策变量Y______'
    # 建立存储决策变量Y的数据结构
    Y = [[[[] for _ in range(0, data.nodeNum)] for _ in range(0,data.nodeNum)] for _ in range(0,data.nodeNum)]  # Y_ijk
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_out_lst:
        for j in data.C_lst:
            if i != j:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        Y[i][j][k] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}_{k}")
    '______添加MTZ辅助变量U______'
    # 建立存储辅助变量U的数据结构
    U = [[] for _ in range(0, data.nodeNum)]  # U_i
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_in_lst:
        U[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=1.0, ub=data.customerNum+2, name=f"U_{i}")
    '______添加卡车时间辅助变量T______'
    # 建立存储辅助变量T的数据结构
    T = [[] for _ in range(0, data.nodeNum)]  # T_i
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_lst:
        T[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{i}")
    '______添加无人机时间辅助变量T_UAV______'
    # 建立存储辅助变量TUAV的数据结构
    TUAV = [[] for _ in range(0, data.nodeNum)]  # TUAV_i
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_lst:
        TUAV[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"TUAV_{i}")
    '______添加访问次序一致变量P______'
    # 建立存储辅助变量P的数据结构
    P = [[[] for _ in range(0, data.nodeNum)] for _ in range(0, data.nodeNum)]  # P_ij
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_out_lst:
        for j in data.C_lst:
            if j != i:
                P[i][j] = m.addVar(vtype=GRB.BINARY, name=f"P_{i}_{j}")

    m.update()
    # var_lst = m.getVars()
    # print(var_lst)

    # 定义目标函数
    obj = LinExpr(0)  # 线性项，初始值为0，可用.addTerms(a,x) 意为将变量x增加到表达式，系数为a
    obj.addTerms(1, T[data.customerNum+1])
    m.setObjective(obj, sense=GRB.MINIMIZE)

    # 定义约束条件:
    # 1.客户点访问约束
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1, X[i][j])
        for i in data.N_out_lst:
            if i != j:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(1, Y[i][j][k])
        num += 1
        m.addConstr(expr == 1, f'C1_{num}')
    # 2.卡车起点车场流出约束
    num = 0
    expr = LinExpr(0)
    for j in data.N_in_lst:
        expr.addTerms(1, X[0][j])
    num += 1
    m.addConstr(expr == 1, f'C2_{num}')
    # 3.卡车终点车场流入约束
    num = 0
    expr = LinExpr(0)
    for i in data.N_out_lst:
        expr.addTerms(1, X[i][data.customerNum+1])
    num += 1
    m.addConstr(expr == 1, f'C3_{num}')
    # 4.卡车在客户点的流平衡约束
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1,X[i][j])
        for k in data.N_in_lst:
            if k != j:
                expr.addTerms(-1,X[j][k])
        num += 1
        m.addConstr(expr == 0, f'C4_{num}')
    # 5.卡车破子圈约束
    num = 0
    for i in data.C_lst:
        for j in data.N_in_lst:
            if i != j:
                expr = LinExpr(0)
                expr.addTerms(1,U[i])
                expr.addTerms(-1,U[j])
                expr.addTerms(data.customerNum+2, X[i][j])
                num += 1
                m.addConstr(expr <= data.customerNum+1, f'C5_{num}')
    # 6.无人机非终点车场发射约束
    num = 0
    for i in data.N_out_lst:
        expr = LinExpr(0)
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(1, Y[i][j][k])
        num += 1
        m.addConstr(expr <= 1,f'C6_{num}')
    # 7.无人机非起点车场回收约束
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(1,Y[i][j][k])
        num += 1
        m.addConstr(expr <= 1, f"C7_{num}")
    # 8.非起点架次下，无人机发射/回收节点的卡车访问约束
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr = LinExpr(0)
                        for h in data.N_out_lst:
                            if h != i:
                                expr.addTerms(1, X[h][i])
                        for l in data.C_lst:
                            if l != k:
                                expr.addTerms(1, X[l][k])
                        expr.addTerms(-2,Y[i][j][k])
                        num += 1
                        m.addConstr(expr >= 0, f'C8_{num}')
    # 9.起点架次下，无人机回收节点的卡车访问约束
    num = 0
    for j in data.C_lst:
        for k in data.N_in_lst:
            if (0,j,k) in data.P_lst:
                expr = LinExpr(0)
                for h in data.N_out_lst:
                    if h != k:
                        expr.addTerms(1,X[h][k])
                expr.addTerms(-1,Y[0][j][k])
                num += 1
                m.addConstr(expr >= 0, f'C9_{num}')
    # 10.无人机架次发射与回收节点卡车访问顺序约束
    num = 0
    for i in data.C_lst:
        for k in data.N_in_lst:
            if k != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[k])
                for j in data.C_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(data.customerNum+2,Y[i][j][k])
                num += 1
                m.addConstr(expr <= data.customerNum+1, f'C10_{num}')
    # 11.无人机发射节点的左侧时间窗约束
    num = 0
    for i in data.C_lst:
        expr = LinExpr(0)
        expr.addTerms(1,T[i])
        expr.addTerms(-1,TUAV[i])
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C11_{num}')
    # 12.无人机发射节点的右侧时间窗约束
    num = 0
    for i in data.C_lst:
        expr = LinExpr(0)
        expr.addTerms(1,TUAV[i])
        expr.addTerms(-1,T[i])
        for j in data.C_lst:
            if j != i:
                for k in data.N_in_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C12_{num}')
    # 13.无人机回收节点的左侧时间窗约束
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        expr.addTerms(1, T[k])
        expr.addTerms(-1, TUAV[k])
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C13_{num}')
    # 14.无人机回收节点的右侧时间窗约束
    num = 0
    for k in data.N_in_lst:
        expr = LinExpr(0)
        expr.addTerms(1, TUAV[k])
        expr.addTerms(-1, T[k])
        for i in data.N_out_lst:
            if i != k:
                for j in data.C_lst:
                    if (i, j, k) in data.P_lst:
                        expr.addTerms(M, Y[i][j][k])
        num += 1
        m.addConstr(expr <= M, f'C14_{num}')
    # 15.卡车有效到达时间约束
    num = 0
    for h in data.N_out_lst:
        for k in data.N_in_lst:
            if h != k:
                expr = LinExpr(0)
                expr.addTerms(1, T[h])
                expr.addTerms(-1, T[k])
                for l in data.C_lst:
                    if l != k:
                        for m_ in data.N_in_lst:
                            if (k,l,m_) in data.P_lst:
                                expr.addTerms(SL, Y[k][l][m_])
                for i in data.N_out_lst:
                    if i != k :
                        for j in data.C_lst:
                            if (i,j,k) in data.P_lst:
                                expr.addTerms(SR,Y[i][j][k])
                expr.addTerms(M,X[h][k])
                num += 1
                m.addConstr(expr + data.costMatrix[h][k] <= M, f'C15_{num}')# 34
    # 16.无人机交付点j的时间约束
    num = 0
    for j in data.C_UAV_lst:
        for i in data.N_out_lst:
            if i != j:
                expr = LinExpr(0)
                expr.addTerms(1,TUAV[i])
                expr.addTerms(-1,TUAV[j])
                for k in data.N_in_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M,Y[i][j][k])
                num += 1
                m.addConstr(expr + UAV_factor * data.costMatrix[i][j] <= M, f'C16_{num}')
    # 17.无人机回收点k的时间约束
    num = 0
    for j in data.C_UAV_lst:
        for k in data.N_in_lst:
            if k != j:
                expr = LinExpr(0)
                expr.addTerms(1,TUAV[j])
                expr.addTerms(-1, TUAV[k])
                for i in data.N_out_lst:
                    if (i,j,k) in data.P_lst:
                        expr.addTerms(M,Y[i][j][k])
                num += 1
                m.addConstr(expr + UAV_factor * data.costMatrix[j][k] + SR <= M, f'C17_{num}')
    # 18.无人机架次电量约束
    num = 0
    for k in data.N_in_lst:
        for j in data.C_lst:
            if j != k:
                for i in data.N_out_lst:
                    if (i,j,k) in data.P_lst:
                        expr = LinExpr(0)
                        expr.addTerms(1,TUAV[k])
                        expr.addTerms(-1, TUAV[j])
                        expr.addTerms(M,Y[i][j][k])
                        num += 1
                        m.addConstr(expr + UAV_factor * data.costMatrix[i][j] <= UAV_endurance + M, f'C18_{num}')
    # 19.客户访问先后辅助变量的定义与建立约束1
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[j])
                expr.addTerms(data.customerNum+2, P[i][j])
                num += 1
                m.addConstr(expr >= 1, f'C19_{num}')
    # 20.客户访问先后辅助变量的定义与建立约束2
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[j])
                expr.addTerms(data.customerNum + 2, P[i][j])
                num += 1
                m.addConstr(expr <= data.customerNum + 1, f'C20_{num}')
    # 21.客户访问先后辅助变量的定义与建立约束3
    num = 0
    for i in data.C_lst:
        for j in data.C_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, P[i][j])
                expr.addTerms(1, P[j][i])
                num += 1
                m.addConstr(expr == 1, f'C21_{num}')
    # 22.无人机架次的先后时间约束/不相交架次下先后时间约束
    num = 0
    for i in data.N_out_lst:
        for k in data.N_in_lst:
            if i != k:
                for l in data.C_lst:
                    if (l != i) and (l != k):
                        expr = LinExpr(0)
                        expr.addTerms(1, TUAV[k])
                        expr.addTerms(-1, TUAV[l])
                        expr.addTerms(M, P[i][l])
                        for j in data.C_lst:
                            if ((i, j, k) in data.P_lst) and (j != l):
                                expr.addTerms(M, Y[i][j][k])
                        for m_ in data.C_lst:
                            if (m_ != i) and (m_ != k) and (m_ != l):
                                for n in data.N_in_lst:
                                    if ((l, m_, n) in data.P_lst) and (n != i) and (n != k):
                                        expr.addTerms(M, Y[l][m_][n])
                        num += 1
                        m.addConstr(expr <= 3*M, f'C22_{num}')
    # 23~.辅助变量符号约束
    m.addConstr(T[0] == 0, f'C23')

    m.addConstr(TUAV[0] == 0, f'C24')

    num = 0
    for j in data.C_lst:
        num += 1
        m.addConstr(P[0][j] == 1, f'C25_{num}')


    # 记录求解开始时间
    start_time = time.time()
    # 求解
    m.optimize()
    m.write('FSTSP.lp')
    if m.status == GRB.OPTIMAL:
        print("-" * 20, "求解成功", '-' * 20)
        # 输出求解总用时
        print(f"求解时间: {time.time() - start_time} s")
        print(f"目标函数为（卡车返回车场的时间）: {m.ObjVal}")
        solution = getSolution(data,m)
        plotSolution(data, solution)
        printSolution(data, solution)
    else:
        print("无解")
    return m


'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\55.问题场景挖掘及综述（二）：FSTSP&PDSTSP建模及其求解\程序代码\data\C101network.txt'  # 这里是节点文件
    customerNum = 9  # C的数量
    UAV_endurance = 20  # 无人机续航
    M = 100  # 约束中存在的M,注意不能取的太大，要保证准确的基础上尽量紧凑
    SR = 0.1  # UAV回收时间
    SL = 0.1  # UAV发射时间
    UAV_factor = 0.5  # 认为无人机的速度是卡车速度的2倍，所以所花的时间是无人机的1/2
    # 读取数据
    data = readData(data_path, customerNum, UAV_endurance)
    print("data.N_lst(N):", data.N_lst)
    print("data.N_out_lst(N0):", data.N_out_lst)
    print("data.N_in_lst(N+):",data.N_in_lst)
    print("data.C_lst(C):", data.C_lst)
    print("data.C_UAV_lst(C'):", data.C_UAV_lst)
    print("data.P_lst(P):", data.P_lst)
    # 输出相关数据
    print("-" * 20, "Problem Information", '-' * 20)
    print(f'节点总数: {data.nodeNum}')
    print(f'客户点总数: {data.customerNum}')
    print(f'可被无人机服务总数: {data.UAVCustomerNum}')
    # 求解
    modelingAndSolve(data)
