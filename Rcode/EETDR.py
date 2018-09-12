import random
import pandas as pd
import tensorly as tl
import numpy as np
import tensorflow as tf
from tensorly.decomposition import _tucker
from tensorly.decomposition._tucker import multi_mode_dot
import build_tensor
import time


class EETDR:

    def __init__(self, TensorX, X, Y, Z, core_G, initU, initI, initA):
        print("初始化算法数据X, Y, Z", file=logfile)
        # 原始UIA构成的张量，ground_true data
        # 采用稀疏张量存储法：矩阵X[m][n]表示第m个非零值的n下标的值（三维张量即是n*3矩阵）。
        # 最后一列X[m][-1]表示该非零值
        self.X = X
        self.TensorX = TensorX
        # UA矩阵，用户对特征的关注度,ground_true data
        self.Y = Y
        # IA矩阵，产品本身特征分布,ground_true data
        self.Z = Z
        self.num_data = X.shape[0]
        self.num_users, self.num_items, self.num_aspects = int(Y.shape[0]), int(Z.shape[0]), int(Y.shape[1])  # 数量
        self.r = 32  # 显式因子维度
        self.r1, self.r2, self.r3 = 64, 64, 32  # 三个分解后矩阵隐式因子的数量
        print("初始化U1，I1，A1，U2，I2，A2", file=logfile)
        # # 初始值U，I，A矩阵由tucker分解一次得到 self.G, factors = tuckerDe.partial_tucker(TensorX1, rank=[self.r + self.r1,
        # self.r + self.r2, self.r + self.r3], n_iter_max=20, verbose=True) self.U, self.I, self.A = factors[0],
        # factors[1], factors[2]
        self.G = core_G
        self.U = initU
        self.I = initI
        self.A = initA
        self.U2 = initU[:, :self.r1]
        self.I2 = initI[:, :self.r2]
        self.A2 = initA[:, :self.r3]
        self.U1 = initU[:, self.r1:]
        self.I1 = initI[:, self.r2:]
        self.A1 = initA[:, self.r3:]
        # 初始值U，I，A矩阵均为随机矩阵
        # self.U1, self.I1, self.A1 = np.random.rand(self.num_users, self.r1), \
        #                             np.random.rand(self.num_items, self.r2), \
        #                             np.random.rand(self.num_aspects, self.r3)
        # self.U2, self.I2, self.A2 = np.random.rand(self.num_users, self.r), \
        #                             np.random.rand(self.num_items, self.r), \
        #                             np.random.rand(self.num_aspects, self.r)
        # self.U, self.I, self.A = np.concatenate((self.U1, self.U2), axis=1), \
        #                          np.concatenate((self.I1, self.I2), axis=1), \
        #                          np.concatenate((self.A1, self.A2), axis=1)
        # self.G = np.random.rand((self.r + self.r1), (self.r + self.r2), (self.r + self.r3))

        print("初始化参数lamda，学习率theta", file=logfile)
        self.lamda = 0.001
        self.lamdax = 0.1
        self.lamday = 0.1
        self.lamda1 = 0.1  # U1，I1，A1的正则化参数
        self.lamda2 = 0.1  # U2，I2，A2的正则化参数
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
        print("稀疏矩阵梯度下降", file=logfile)
        ##############################################################################################
        # 停止条件是所有梯度放缓，小于学习率lamda
        # U2的随机梯度下降更新########################################################################
        i = random.randint(0, self.num_users - 1)  # 随机梯度下降， 随机选取第i个用户更新
        for a in range(self.r1):  # 对i用户的向量每个值进行元素级更新
            print("U2[{}][{}]  = {}".format(i, a, U2[i][a]), file=logfile)
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if i == self.X[data][0]:
                    j = self.X[data][1]
                    k = self.X[data][2]
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for b1 in range(rr2):
                        var3 = self.I[j][b1]
                        for c1 in range(rr3):
                            temp = self.G[a][b1][c1] * var3 * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * U2[i][a]  # 完全负梯度
            U2[i][a] = U2[i][a] + self.theta1 * neg_gradient  # Uia更新
            print("U2[{}][{}]  = {}".format(i, a, U2[i][a]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        ##############################################################################################
        # I2的随机梯度下降更新########################################################################
        j = random.randint(0, self.num_items - 1)  # 随机选取第j个产品更新
        for b in range(self.r2):  # 对产品j的向量每个值进行元素级更新
            print("I2[{}][{}]  = {}".format(j, b, I2[j][b]), file=logfile)
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if j == self.X[data][1]:
                    i = self.X[data][0]
                    k = self.X[data][2]
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1 in range(rr1):
                        var3 = self.U[i][a1]
                        for c1 in range(rr3):
                            temp = self.G[a1][b][c1] * var3 * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * I2[j][b]  # 完全负梯度
            I2[j][b] = I2[j][b] + self.theta2 * neg_gradient  # Ijb更新
            print("I2[{}][{}]  = {}".format(j, b, I2[j][b]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        ##############################################################################################
        # A2的随机梯度下降更新########################################################################
        k = random.randint(0, self.num_aspects - 1)  # 随机选取第k个特征aspect更新
        for c in range(self.r3):  # 对aspect k的向量每个值进行元素级更新
            print("A2[{}][{}]  = {}".format(k, c, A2[k][c]))
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if k == self.X[data][2]:
                    i = self.X[data][0]
                    j = self.X[data][1]
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1 in range(rr1):
                        var3 = self.U[i][a1]
                        for b1 in range(rr2):
                            temp = self.G[a1][b1][c] * self.I[j][b1] * var3
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2
            neg_gradient = sum1 - self.lamda2 * A2[k][c]  # 完全负梯度
            A2[k][c] = A2[k][c] + self.theta3 * neg_gradient  # Akc更新
            print("A2[{}][{}]  = {}".format(k, c, A2[k][c]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)
        ##############################################################################################
        # U1,I1,A1的随机梯度下降更新##################################################################
        k = random.randint(0, self.num_aspects - 1)
        i_pie = random.randint(0, self.num_users - 1)  # 随机选取第i个用户更新
        j_pie = random.randint(0, self.num_items - 1)  # 随机选取第j个产品更新
        k_pie = random.randint(0, self.num_aspects - 1)  # 随机选取第k个aspect更新
        for a in range(self.r):  # 对i用户的向量每个值进行元素级更新
            print("U1[{}][{}]  = {}".format(i_pie, a, U1[i_pie][a]), file=logfile)
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if i_pie == self.X[data][0]:
                    j = self.X[data][1]
                    k = self.X[data][2]
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i_pie][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for b1 in range(rr2):
                        var3 = self.I[j][b1]
                        for c1 in range(rr3):
                            temp = self.G[a + self.r1][b1][c1] * var3 * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2
            sum3 = 0
            for o in range(self.r):
                sum3 = sum3 + U1[i_pie][o] * A1[k_pie][o]
            explicit_gradient1 = self.lamdax * (self.Y[i_pie][k_pie] - sum3) * A1[k_pie][a]
            # 完全负梯度
            neg_gradient = sum1 - self.lamda2 * U1[i_pie][a] + explicit_gradient1
            U1[i_pie][a] = U1[i_pie][a] + self.theta1 * neg_gradient  # Uia更新
            print("U1[{}][{}]  = {}".format(i_pie, a, U1[i_pie][a]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        for b in range(self.r):  # 对产品j的向量每个值进行元素级更新
            print("I1[{}][{}]  = {}".format(j_pie, b, I1[j_pie][b]), file=logfile)
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if j_pie == self.X[data][1]:
                    i = self.X[data][0]
                    k = self.X[data][2]
                    X_ijk = self.X[data][3]
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j_pie][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1 in range(rr1):
                        var3 = self.U[i][a1]
                        for c1 in range(rr3):
                            temp = self.G[a1][b + self.r][c1] * var3 * self.A[k][c1]
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2
            sum3 = 0
            for o in range(self.r):
                sum3 = sum3 + I1[j_pie][o] * A1[k_pie][o]
            explicit_gradient2 = self.lamday * (self.Y[i_pie][k_pie] - sum3) * A1[k_pie][a]
            neg_gradient = sum1 - self.lamda2 * I1[j_pie][b] + explicit_gradient2  # 完全负梯度
            I1[j_pie][b] = I1[j_pie][b] + self.theta2 * neg_gradient  # Ijb更新
            print("I1[{}][{}]  = {}".format(j_pie, b, I1[j_pie][b]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)
        for c in range(self.r):  # 对aspect k的向量每个值进行元素级更新
            print("A1[{}][{}]  = {}".format(k_pie, c, A1[k_pie][c]), file=logfile)
            sum1 = 0  # 稀疏张量分解得到的负梯度
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):  # 稀疏张量的取下标。
                if k_pie == self.X[data][2]:
                    i = self.X[data][0]
                    j = self.X[data][1]
                    X_ijk = self.X[data][3]
                    print(i, j, k, X_ijk, file=logfile)
                    X2_ijk = 0
                    rr1 = self.r + self.r1
                    rr2 = self.r + self.r2
                    rr3 = self.r + self.r3
                    for a0 in range(rr1):
                        var1 = self.U[i][a0]
                        for b0 in range(rr2):
                            var2 = self.I[j][b0]
                            for c0 in range(rr3):
                                X2_ijk = X2_ijk + self.G[a0][b0][c0] * var1 * var2 * self.A[k_pie][c0]
                    e_ijk = X_ijk - X2_ijk
                    sum2 = 0
                    for a1 in range(rr1):
                        var3 = self.U[i][a1]
                        for b1 in range(rr2):
                            temp = self.G[a1][b1][c + self.r] * self.I[j][b1] * var3
                            sum2 += temp
                    sum1 = sum1 + self.lamda * e_ijk * sum2

            sum3 = 0
            sum4 = 0
            for o in range(self.r):
                sum3 = sum3 + U1[i_pie][o] * A1[k_pie][o]
                sum4 = sum4 + I1[j_pie][o] * A1[k_pie][o]

            explicit_gradient3 = self.lamdax * (self.Y[i_pie][k_pie] - sum3) * U1[i_pie][c]
            explicit_gradient4 = self.lamday * (self.Z[j_pie][k_pie] - sum3) * I1[j_pie][c]
            neg_gradient = sum1 - self.lamda2 * A1[k_pie][c] + explicit_gradient3 + explicit_gradient4  # 完全负梯度
            A1[k_pie][c] = A1[k_pie][c] + self.theta3 * neg_gradient  # Akc更新
            print("A1[{}][{}]  = {}".format(k_pie, c, A1[k_pie][c]), file=logfile)
            print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        self.U1 = U1
        self.U2 = U2
        self.I1 = I1
        self.I2 = I2
        self.A1 = A1
        self.A2 = A2
        #################################################################################################
        ######更新核张量G
        self.U, self.I, self.A = np.concatenate((self.U1, self.U2), axis=1), \
                                 np.concatenate((self.I1, self.I2), axis=1), \
                                 np.concatenate((self.A1, self.A2), axis=1)
        factors = [self.U, self.I, self.A]
        rec_errors = []

        modes1 = list(range(_tucker.T.ndim(self.TensorX)))
        for index, mode in enumerate(modes1):
            self.G = _tucker.multi_mode_dot(self.TensorX, factors, modes=modes1, transpose=True)
        modes2 = list(range(_tucker.T.ndim(self.G)))
        for index, mode in enumerate(modes2):
            TensorX_approximation = _tucker.multi_mode_dot(self.G, factors, modes=modes2, transpose=False)

        norm_tensor = _tucker.T.norm(self.TensorX, 2)
        rec_error = _tucker.sqrt(abs(norm_tensor ** 2 - _tucker.T.norm(TensorX_approximation, 2) ** 2)) / norm_tensor
        a = _tucker.sqrt(abs(_tucker.T.norm(self.Y, 2) ** 2 - _tucker.T.norm((U1.dot(A1.transpose())), 2) ** 2))
        b = _tucker.sqrt(abs(_tucker.T.norm(self.Z, 2) ** 2 - _tucker.T.norm((I1.dot(A1.transpose())), 2) ** 2))
        print(rec_error, a, b)
        rec_errors.append(rec_error)
        print("Once OK，rec={}".format(rec_errors))

        return rec_errors


if __name__ == "__main__":
    # Bt = build_tensor.Build_tensor(infile="C:/Users/Syd/OneDrive/Work/EETDR/data/yelp_recursive_train.entry",
    #                                outfile="E:/PYworkspace/EETDR/result/UIA.csv")
    # X = Bt.build_tensor(tensor=False, matix=True, sprasefile=False)
    # Y = Bt.build_UA()
    # Z = Bt.build_IA()
    # X = pd.read_csv("E:/PYworkspace/EETDR/result/UIA.csv", dtype="float32", header=None, names=["user_index", "item_index", "aspect_index", "value"])
    # print(X.iloc[1].user_index)
    # 文件加载初始值##################################################
    X1 = np.load("E:/PYworkspace/EETDR/result/X3000.npy")
    Y1 = np.load("E:/PYworkspace/EETDR/result/UA3000.npy")
    Z1 = np.load("E:/PYworkspace/EETDR/result/IA3000.npy")
    TensorX1 = np.load("E:/PYworkspace/EETDR/result/TensorX.npy")
    core_G = np.load("E:/PYworkspace/EETDR/result/core.npy")
    initU = np.load("E:/PYworkspace/EETDR/result/initU.npy")
    initI = np.load("E:/PYworkspace/EETDR/result/initI.npy")
    initA = np.load("E:/PYworkspace/EETDR/result/initA.npy")
    logfile = open("E:\PYworkspace\EETDR\\result\log.txt", "w")
    #################################################################
    print(initA.shape)
    fine = EETDR(TensorX1, X1, Y1, Z1, core_G, initU, initI, initA)

    print("用户数{}".format(fine.num_users))
    print("产品数{}".format(fine.num_items))
    print("Aspect数{}".format(fine.num_aspects))
    print("非零数据条数{}".format(fine.num_data))
    print("显式特征向量长度{}".format(fine.r))
    print("隐式特征向量长度{} {} {}".format(fine.r1, fine.r2, fine.r3))
    print("非零数据条数{}".format(fine.num_data))
    flag = True
    while flag:
        rec_errors = fine.eetdr_SGDonce()
        if rec_errors[-1] < 10e-5:
            flag = False
