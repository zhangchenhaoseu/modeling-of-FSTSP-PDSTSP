# 靡不有初，鲜克有终
# 开发时间：2023/12/7 9:34
''' FSTSP,模型服从以下假设：
a.单车场
b.单辆卡车
c.单架无人机/多架无人机：模型认为是单架无人机，因为是对称的
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

        self.UAVCustomerNuminDCRange = 0  # 可被无人机访问的客户、同时与车场的距离在无人机续航范围内的总数量
        self.UAVCustomerinDCRange_lst = []  # 存放所有可被无人机访问的客户id的列表,同时还在无人机运行的范围内 C''∈ C'


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
            data.C_UAV_lst.append(data_df.loc[i, 'CUST NO']-1)  # 存储所有可以被无人机访问的客户点,从0开始的
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

    # 找到距离车场在无人机续航范围内的、可被无人机访问的客户点
    for i in data.C_UAV_lst:
        if UAV_factor*(data.costMatrix[0][i] + data.costMatrix[i][0]) <= UAV_endurance:
            data.UAVCustomerinDCRange_lst.append(i)
    data.UAVCustomerNuminDCRange = len(data.UAVCustomerinDCRange_lst)
    # print("costMatrix:")
    # print(data.costMatrix)
    return data


'''定义解类。建立求解结果的数据结构框架，并建立将Gurobi求解结果存放在数据结构中的连接'''
class Solution:
    ObjVal = 0  # 目标函数值
    X = None  # 用于存放卡车的决策变量X_ij的值
    X_routes = None  # 用于存放卡车所经过的节点序列
    Y = None  # 用于存放无人机的决策变量Y_i的值
    Y_routes = None  # 用于存放无人机所经过的节点序列
    X_routesNum = 0  # 卡车路径数量
    Y_routesNum = 0  # 无人机架次数量

    def __init__(self, data, model):  # 建立类属性，解类的建立需要输入数据和模型
        self.ObjVal = model.ObjVal
        self.X = [[0 for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)]  # 由于仅作存储作用方便调用，因此此处全连接即可
        self.X_routes = []
        self.Y = [0 for k in range(0, data.nodeNum)]
        self.Y_routes = []
        self.U = [0 for k in range(0, data.nodeNum)]



def getSolution(data, model):  # 定义函数，从Gurobi输出的模型中获得目标函数值、决策变量X和Y、继而分别得到卡车和无人机路由的节点序列
    solution = Solution(data, model)
    # 在三维矩阵结构中存储自变量的取值，拆字段，检验是否是x，然后保存
    for v in model.getVars():
        split_arr = re.split(r"_", v.VarName)  # 将gurobi形式的变量进行拆解，便于利用数据结构实现实现存储
        if split_arr[0] == 'X' and v.x != 0:
            # print(v)
            solution.X[int(split_arr[1])][int(split_arr[2])] = v.x  # X_ij
        elif split_arr[0] == 'Y' and v.x != 0:
            solution.Y[int(split_arr[1])]= v.x  # Y_i
        elif split_arr[0] == 'U' and v.x != 0:
            solution.U[int(split_arr[1])] = v.x  # U_i
        else:
            pass
            # print(v)
    print("卡车路由变量 solution.X:", solution.X)
    print("无人机路由变量 solution.Y:", solution.Y)
    print("卡车访问辅助变量 solution.U:", solution.U)

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
        if round(solution.Y[i]) != 0:  # 取最接近的整数，因为有的量虽然非零，但很小
            solution.Y_routes.append((0, i, data.customerNum+1))
            solution.Y_routesNum += 1  # 无人机架次数
    print('无人机的飞行路线 solution.Y_routes:', solution.Y_routes)

    return solution


'''绘图。以图像展示VRP求解结果'''
def plotSolution(data, solution):
    # 绘制画布
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{data.UAVCustomerNum} UAV-eligible and {data.UAVCustomerNuminDCRange} Customers within UAV's range from DC in total {data.customerNum} Customers,single truck")
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
            point_start = (data.corX[a], data.corY[a])
            point_end = (data.corX[b], data.corY[b])
            draw_curved_line(point_start, point_end, curvature)
            draw_curved_line(point_end, point_start, -curvature)
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
    m = Model('PDSTSP')

    # 模型设置：由于存在函数printSolution，因此关闭输出;以及容许误差
    m.setParam('MIPGap', 0.1)
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
    Y = [[] for _ in range(0, data.nodeNum)]  # Y_i
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.UAVCustomerinDCRange_lst:
        Y[i] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}")
    '______添加MTZ辅助变量U______'
    # 建立存储辅助变量U的数据结构
    U = [[] for _ in range(0, data.nodeNum)]  # U_i
    # 根据数据结构向模型中添加对应下标的变量
    for i in data.N_in_lst:
        U[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=1.0, ub=data.customerNum+2, name=f"U_{i}")
    '______添加辅助变量Z______'
    Z = m.addVar(vtype=GRB.CONTINUOUS,name='Z')
    m.update()
    var_lst = m.getVars()
    # print(var_lst)

    # 定义目标函数,minZ
    obj = LinExpr(0)  # 线性项，初始值为0，可用.addTerms(a,x) 意为将变量x增加到表达式，系数为a
    obj.addTerms(1, Z)
    m.setObjective(obj, sense=GRB.MINIMIZE)

    # 定义约束条件:
    # 1.卡车路由时间最短约束
    num = 0
    expr = LinExpr(0)
    for i in data.N_out_lst:
        for j in data.N_in_lst:
            if i != j:
                expr.addTerms(data.costMatrix[i][j], X[i][j])
    num += 1
    m.addConstr(expr <= Z, f'C1_{num}')
    # 2.无人机路由时间最短约束
    num = 0
    expr = LinExpr(0)
    for i in data.UAVCustomerinDCRange_lst:  # c''
        expr.addTerms(UAV_factor*(data.costMatrix[0][i]+data.costMatrix[i][data.customerNum+1]), Y[i])
    num += 1
    m.addConstr(expr <= Z, f'C2_{num}')
    # 3.客户点访问约束
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1, X[i][j])
        if j in data.UAVCustomerinDCRange_lst:
            expr.addTerms(1, Y[j])
        num += 1
        m.addConstr(expr == 1, f'C3_{num}')
    # 4.卡车起点车场流出约束
    num = 0
    expr = LinExpr(0)
    for j in data.N_in_lst:
        expr.addTerms(1, X[0][j])
    num += 1
    m.addConstr(expr == 1, f'C4_{num}')
    # 5.卡车起点车场流入约束
    num = 0
    expr = LinExpr(0)
    for i in data.N_out_lst:
        expr.addTerms(1, X[i][data.customerNum+1])
    num += 1
    m.addConstr(expr == 1, f'C5_{num}')
    # 6.卡车客户节点流平衡约束
    num = 0
    for j in data.C_lst:
        expr = LinExpr(0)
        for i in data.N_out_lst:
            if i != j:
                expr.addTerms(1, X[i][j])
        for k in data.N_in_lst:
            if k !=j:
                expr.addTerms(-1, X[j][k])
        num += 1
        m.addConstr(expr == 0, f'C6_{num}')
    # 7.卡车破子圈约束
    num = 0
    for i in data.C_lst:
        for j in data.N_in_lst:
            if j != i:
                expr = LinExpr(0)
                expr.addTerms(1, U[i])
                expr.addTerms(-1, U[j])
                expr.addTerms((data.customerNum+2),X[i][j])
                num += 1
                m.addConstr(expr <= data.customerNum+1,f'C7_{num}')

    # 记录求解开始时间
    start_time = time.time()
    # 求解
    m.optimize()
    m.write('PDSTSP.lp')
    if m.status == GRB.OPTIMAL:
        print("-" * 20, "求解成功", '-' * 20)
        # 输出求解总用时
        print(f"求解时间: {time.time() - start_time} s")
        print(f"目标函数为: {m.ObjVal}")
        solution = getSolution(data,m)
        plotSolution(data, solution)
        printSolution(data, solution)
    else:
        print("无解")
    return m


def draw_curved_line(point1, point2, curvature):  # 绘制两点之间的曲线，用于无人机路径图示 point1→point2
    x1, y1 = point1
    x2, y2 = point2
    # 计算中点
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    # 计算法线斜率,处理垂直线的情况
    if x2 - x1 == 0:
        normal_slope = 999
    else:
        normal_slope = -1 / ((y2 - y1) / (x2 - x1))
    # 计算偏移量
    offset = curvature * np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
    # 计算控制点
    control_x = mid_x + offset / np.sqrt(1 + normal_slope**2)
    control_y = mid_y + normal_slope * (control_x - mid_x)
    # 生成曲线点
    t = np.linspace(0, 1, 100)
    curve_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * control_x + t**2 * x2
    curve_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * control_y + t**2 * y2
    # 绘制曲线
    plt.plot(curve_x, curve_y, color='red', linewidth=1, linestyle=':')



'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\55.问题场景挖掘及综述（二）：FSTSP&PDSTSP建模及其求解\程序代码\data\C101network.txt'  # 这里是节点文件
    customerNum = 9 # C的数量
    UAV_endurance = 20  # 无人机续航
    M = 100  # 约束中存在的M
    SR = 0.1  # UAV回收时间
    SL = 0.1  # UAV发射时间
    UAV_factor = 0.5  # 认为无人机的速度是卡车速度的2倍，所以所花的时间是无人机的1/2
    curvature = 0.05   # 设置无人机图示中的曲率
    # 读取数据
    data = readData(data_path, customerNum, UAV_endurance)
    print("data.N_lst(N):", data.N_lst)
    print("data.N_out_lst(N0):", data.N_out_lst)
    print("data.N_in_lst(N+):",data.N_in_lst)
    print("data.C_lst(C):", data.C_lst)
    print("data.C_UAV_lst(C'):", data.C_UAV_lst)
    # 输出相关数据
    print("-" * 20, "Problem Information", '-' * 20)
    print(f'节点总数: {data.nodeNum}')
    print(f'客户点总数: {data.customerNum}')
    print(f'可被无人机服务总数: {data.UAVCustomerNum}')
    # 求解
    modelingAndSolve(data)
