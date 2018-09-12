import random
import pandas as pd
import tensorly as tl
import numpy as np
import torch
import tensorflow as tf
from tensorly.decomposition import tucker
from tensorly.decomposition._tucker import multi_mode_dot
import build_tensor


class EETDR:

    def __init__(self, X, Y, Z):
        print("初始化算法数据X, Y, Z")
        # 原始UIA构成的张量，ground_true data
        # 采用稀疏张量存储法：矩阵X[m][n]表示第m个非零值的n下标的值（三维张量即是n*3矩阵）。
        # 最后一列X[m][-1]表示该非零值
        self.X = X
        # UA矩阵，用户对特征的关注度,ground_true data
        self.Y = Y
        # IA矩阵，产品本身特征分布,ground_true data
        self.Z = Z
        self.num_data = X.shape[0]
        self.num_users, self.num_items, self.num_aspects = int(Y.shape[0]), int(Z.shape[0]), int(Y.shape[1])  # 数量
        self.r = 128  # 显式因子维度
        self.r1, self.r2, self.r3 = 200, 200, 100  # 三个分解后矩阵隐式因子的数量
        # # 初始值U，I，A矩阵由tucker分解一次得到
        print("初始化U1，I1，A1，U2，I2，A2")
        # self.G, factors = tucker(X_tensor, (self.r + self.r1, self.r + self.r2, self.r + self.r3), n_iter_max=1)
        # self.U, self.I, self.A = factors[0], factors[1], factors[2]
        # 初始值U，I，A矩阵均为随机矩阵
        self.U1, self.I1, self.A1 = torch.random.rand(self.num_users, self.r1), \
                                    torch.random.rand(self.num_items, self.r2), \
                                    torch.random.rand(self.num_aspects, self.r3)
        self.U2, self.I2, self.A2 = torch.random.rand(self.num_users, self.r), \
                                    torch.random.rand(self.num_items, self.r), \
                                    torch.random.rand(self.num_aspects, self.r)
        self.U, self.I, self.A = torch.concatenate((self.U1, self.U2), axis=1), \
                                 torch.concatenate((self.I1, self.I2), axis=1), \
                                 torch.concatenate((self.A1, self.A2), axis=1)
        self.G = np.random.rand((self.r+self.r1), (self.r+self.r2), (self.r+self.r3))
        print("初始化参数lamda，学习率theta")
        self.lamda = 1e-10
        self.lamdax = 0.1
        self.lamday = 0.1
        self.lamda1 = 0.1  # U1，I1，A1的正则化参数
        self.lamda2 = 1  # U2，I2，A2的正则化参数
        self.lamda3 = 0.1  # 核张量G的正则化参数
        self.theta1 = 0.1  # U2随机梯度下降学习率
        self.theta2 = 0.1  # I2随机梯度下降学习率
        self.theta3 = 0.1  # A2随机梯度下降学习率
        self.theta4 = 0.1
        self.theta5 = 0.1
        self.theta6 = 0.1

        return

    def eetdr_SGDonce(self):
        U1 = self.U1
        U2 = self.U2
        I1 = self.I1
        I2 = self.I2
        A1 = self.A1
        A2 = self.A2
        print("ALS稀疏矩阵梯度下降")
        # U2的随机梯度下降更新########################################################################
        i = random.randint(0, self.num_users - 1)  # 随机梯度下降， 随机选取第i个用户更新
        for a in range(self.r1):  # 对i用户的向量每个值进行元素级更新
            print("U2[{}][{}]  = {}".format(i, a, U2[i][a]))
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if i == self.X[data][0]:
                    j = self.X[data][1]
                    k = self.X[data][2]
                    print(i, j, k)
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    for a0 in range(self.r + self.r1):
                        for b0 in range(self.r + self.r2):
                            for c0 in range(self.r + self.r3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for b1 in range(self.r + self.r2):
                        for c1 in range(self.r + self.r3):
                            temp = self.G[a][b1][c1] * self.I[j][b1] * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda*e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * U2[i][a]  # 完全负梯度
            print("完全负梯度{}".format(neg_gradient))
            U2[i][a] = U2[i][a] + self.theta1 * neg_gradient  # Uia更新
            print("U2[{}][{}]  = {}".format(i, a, U2[i][a]))
        ##############################################################################################
        # I2的随机梯度下降更新########################################################################
        j = random.randint(0, self.num_items-1)  # 随机选取第j个产品更新
        for b in range(self.r2):  # 对产品j的向量每个值进行元素级更新
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if j == self.X[data][1]:
                    i = self.X[data][0]
                    k = self.X[data][2]
                    X_ijk = self.V[data]
                    X2_ijk = 0
                    for a0 in range(self.r + self.r1):
                        for b0 in range(self.r + self.r2):
                            for c0 in range(self.r + self.r3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1 in range(self.r + self.r1):
                        for c1 in range(self.r + self.r3):
                            temp = self.G[a1][b][c1] * self.U[i][a1] * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda*e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * I2[j][b]  # 完全负梯度
            I2[j][b] = I2[j][b] + self.theta2 * neg_gradient  # Ijb更新
        ##############################################################################################
        # A2的随机梯度下降更新########################################################################
        ff, cc = A2.shape
        k = random.random(0, ff)  # 随机选取第k个特征aspect更新
        for c in range(cc):  # 对aspect k的向量每个值进行元素级更新
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if k == self.X[data][2]:
                    i = self.X[data][0]
                    j = self.X[data][1]
                    X_ijk = self.V[data]
                    X2_ijk = 0
                    for a0, b0, c0 in range(self.r + self.r1), range(self.r + self.r2), range(self.r + self.r3):
                        X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1, b1 in range(self.r + self.r1), range(self.r + self.r2):
                        temp = self.G[i][j][k] * self.I[j][b1] * self.U[i][a1]
                        sum2 += temp
                    sum1 = sum1 + e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * A2[k][c]  # 完全负梯度
            A2[k][c] = A2[k][c] + self.theta3 * neg_gradient  # Akc更新
        ##############################################################################################
        # U1,I1,A1的随机梯度下降更新##################################################################
        uu, aa = U1.shape
        ii, bb = I1.shape
        ff, cc = A1.shape
        i_pie = random.random(0, uu)  # 随机选取第i个用户更新
        j_pie = random.random(0, ii)  # 随机选取第j个产品更新
        k_pie = random.random(0, ff)  # 随机选取第k个aspect更新
        for a in range(aa):  # 对i用户的向量每个值进行元素级更新
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if i_pie == self.X[data][0]:
                    j = self.X[data][1]
                    k = self.X[data][2]
                    X_ijk = self.V[data]
                    X2_ijk = 0
                    for a0, b0, c0 in range(self.r + self.r1), range(self.r + self.r2), range(self.r + self.r3):
                        X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i_pie][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for b1, c1 in range(self.r + self.r2), range(self.r + self.r3):
                        temp = self.G[a + self.r1][b1][c1] * self.I[j][b1] * self.A[k][c1]
                        sum2 += temp
                    sum1 = sum1 + e_ijk * sum2
            sum3 = 0
            for o in range(self.r):
                sum3 = sum3 + U1[i_pie][o] * A1[k_pie][o]
            explicit_gradient1 = self.lamdax * (self.Y[i_pie][k_pie] - sum3) * A1[k_pie][a]
            # 完全负梯度
            neg_gradient = sum1 - self.lamda2 * U1[i_pie][a] + explicit_gradient1
            U1[i_pie][a] = U1[i_pie][a] + self.theta1 * neg_gradient  # Uia更新

        for b in range(bb):  # 对产品j的向量每个值进行元素级更新
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if j_pie == self.X[data][1]:
                    i = self.X[data][0]
                    k = self.X[data][2]
                    X_ijk = self.V[data]
                    X2_ijk = 0
                    for a0, b0, c0 in range(self.r + self.r1), range(self.r + self.r2), range(self.r + self.r3):
                        X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1, c1 in range(self.r + self.r1), range(self.r + self.r3):
                        temp = self.G[a1][b][c1] * self.U[i][a1] * self.A[k][c1]
                        sum2 += temp
                    sum1 = sum1 + e_ijk * sum2
            sum3 = 0
            for o in range(self.r):
                sum3 = sum3 + I1[j_pie][o] * A1[k_pie][o]
            explicit_gradient2 = self.lamday * (self.Y[i_pie][k_pie] - sum3) * A1[k_pie][a]
            neg_gradient = sum1 - self.lamda2 * I2[j][b] + explicit_gradient2  # 完全负梯度
            I2[j][b] = I2[j][b] + self.theta2 * neg_gradient  # Ijb更新

        for c in range(cc):  # 对aspect k的向量每个值进行元素级更新
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if k_pie == self.X[data][2]:
                    i = self.X[data][0]
                    j = self.X[data][1]
                    X_ijk = self.V[data]
                    X2_ijk = 0
                    for a0, b0, c0 in range(self.r + self.r1), range(self.r + self.r2), range(self.r + self.r3):
                        X2_ijk = X2_ijk + self.G[a0][b0][c0] * self.U[i][a0] * self.I[j][b0] * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1, b1 in range(self.r + self.r1), range(self.r + self.r2):
                        temp = self.G[i][j][k] * self.I[j][b1] * self.U[i][a1]
                        sum2 += temp
                    sum1 = sum1 + e_ijk * sum2

            sum3 = 0
            sum4 = 0
            for o in range(self.r):
                sum3 = sum3 + U1[i_pie][o] * A1[k_pie][o]
                sum4 = sum4 + I1[j_pie][o] * A1[k_pie][o]

            explicit_gradient3 = self.lamdax * (self.U[i_pie][k_pie] - sum3) * U1[i_pie][c]
            explicit_gradient4 = self.lamday * (self.Z[j_pie][k_pie] - sum3) * I1[j_pie][c]
            neg_gradient = sum1 - self.lamda2 * A2[k][c] + explicit_gradient3 + explicit_gradient4  # 完全负梯度
            A2[k][c] = A2[k][c] + self.theta3 * neg_gradient  # Akc更新

        return U1, U2, I1, I2, A1, A2


if __name__ == "__main__":
    # Bt = build_tensor.Build_tensor(infile="C:/Users/Syd/OneDrive/Work/EETDR/data/yelp_recursive_train.entry",
    #                                outfile="E:/PYworkspace/EETDR/result/UIA.csv")
    # X = Bt.build_tensor(tensor=False, matix=True, sprasefile=False)
    # Y = Bt.build_UA()
    # Z = Bt.build_IA()
    # X = pd.read_csv("E:/PYworkspace/EETDR/result/UIA.csv", dtype="float32", header=None, names=["user_index", "item_index", "aspect_index", "value"])
    # print(X.iloc[1].user_index)
    # 测试用##################################################
    X = np.load("E:/PYworkspace/EETDR/result/X3000.npy")
    Y = np.load("E:/PYworkspace/EETDR/result/UA3000.npy")
    Z = np.load("E:/PYworkspace/EETDR/result/IA3000.npy")
    ##########################################################
    fine = EETDR(X, Y, Z)
    print("用户数{}".format(fine.num_users))
    print("产品数{}".format(fine.num_items))
    print("Aspect数{}".format(fine.num_aspects))
    print("非零数据条数{}".format(fine.num_data))
    print("显式特征向量长度{}".format(fine.r))
    print("隐式特征向量长度{} {} {}".format(fine.r1, fine.r2, fine.r3))
    print("非零数据条数{}".format(fine.num_data))
    fine.eetdr_SGDonce()
