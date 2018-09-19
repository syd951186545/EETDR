"""
minibatch and less reconstruction support a faster compute
"""

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
        print("初始化算法数据X, Y, Z")
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
        print("初始化U1，I1，A1，U2，I2，A2")
        # 初始值U，I，A矩阵由tucker分解一次得到
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

        print("初始化参数lamda，学习率theta")
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
        self.rec_errors = []
        return

    def eetdr_SGDonce(self, TensorX_approximation):

        U = self.U
        I = self.I
        A = self.A
        U1 = self.U1
        U2 = self.U2
        I1 = self.I1
        I2 = self.I2
        A1 = self.A1
        A2 = self.A2

        print("稀疏矩阵小批量(10)梯度下降", file=logfile)
        for minibatch in range(10):
            print("小批量次数{}".format(minibatch))
            ##############################################################################################
            # U2的随机梯度下降更新########################################################################
            i_once = random.randint(0, self.num_users - 1)  # 随机梯度下降， 随机选取第i个用户更新
            j_once = random.randint(0, self.num_items - 1)  # 随机梯度下降， 随机选取第i个用户更新
            k_once = random.randint(0, self.num_aspects - 1)  # 随机梯度下降， 随机选取第i个用户更新
            rr1 = self.r + self.r1
            rr2 = self.r + self.r2
            rr3 = self.r + self.r3
            # e_ijk = self.TensorX[i_once][j_once][k_once]-TensorX_approximation[i_once][j_once][k_once]
            Ur = self.r1 + self.r
            # 对i用户的向量每个元素进行元素级更新
            for a in range(Ur):
                print("updata U[{}][{}]".format(i_once, a))
                # print("U[{}][{}]  = {}".format(i_once, a, U[i_once][a]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if i_once == self.X[data][0]:
                        j = self.X[data][1]
                        k = self.X[data][2]
                        e_ijk = self.TensorX[i_once][j][k] - TensorX_approximation[i_once][j][k]
                        sum2 = 0
                        for b1 in range(rr2):
                            var3 = self.I[j][b1]
                            for c1 in range(rr3):
                                temp = self.G[a][b1][c1] * var3 * self.A[k][c1]
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + self.lamda * e_ijk * sum2
                if a < self.r1:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * U2[i_once][a]
                    # U2ia更新
                    self.U2[i_once][a] = U2[i_once][a] + self.theta1 * neg_gradient

                    # print("U2[{}][{}]  = {}".format(i_once, a, U2[i_once][a]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)
                else:
                    sum3 = 0
                    for o in range(self.r):
                        sum3 = sum3 + U1[i_once][o] * A1[k_once][o]
                    explicit_gradient1 = (self.Y[i_once][k_once] - sum3) * A1[k_once][a - self.r1]
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * U1[i_once][
                        a - self.r1] + self.lamdax * explicit_gradient1
                    # U1ia更新
                    self.U1[i_once][a - self.r1] = U1[i_once][a - self.r1] + self.theta1 * neg_gradient
                    # print("U1[{}][{}]  = {}".format(i_once, a - self.r1, U1[i_once][a - self.r1]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

            ##############################################################################################
            # I2的随机梯度下降更新########################################################################
            Ir = self.r2 + self.r
            # 对产品j的向量每个值进行元素级更新
            for b in range(Ir):
                print("updata I[{}][{}]".format(j_once, b))
                # print("I[{}][{}]  = {}".format(j_once, b, I[j_once][b]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if j_once == self.X[data][1]:
                        i = self.X[data][0]
                        k = self.X[data][2]
                        e_ijk = self.TensorX[i][j_once][k] - TensorX_approximation[i][j_once][k]
                        sum2 = 0
                        for a1 in range(rr1):
                            var3 = self.U[i][a1]
                            for c1 in range(rr3):
                                temp = self.G[a1][b][c1] * var3 * self.A[k][c1]
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + self.lamda * e_ijk * sum2
                if b < self.r2:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * I2[j_once][b]
                    # I2jb更新
                    self.I2[j_once][b] = I2[j_once][b] + self.theta2 * neg_gradient
                    # print("I2[{}][{}]  = {}".format(j_once, b, I2[j_once][b]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

                else:
                    sum3 = 0
                    for o in range(self.r):
                        sum3 = sum3 + I1[j_once][o] * A1[k_once][o]
                    explicit_gradient2 = self.lamday * (self.Y[j_once][k_once] - sum3) * A1[k_once][b - self.r2]
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * I1[j_once][b - self.r2] + explicit_gradient2
                    # I1jb更新
                    self.I1[j_once][b - self.r2] = I1[j_once][b - self.r2] + self.theta2 * neg_gradient
                    # print("I1[{}][{}]  = {}".format(j_once, b - self.r2, I1[j_once][b - self.r2]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

            ##############################################################################################
            # A2的随机梯度下降更新########################################################################
            Ar = self.r3 + self.r
            # 对aspect k的向量每个值进行元素级更新
            for c in range(Ar):
                print("updata A[{}][{}]".format(k_once, c))
                # print("A[{}][{}]  = {}".format(k_once, c, A[k_once][c]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if k_once == self.X[data][2]:
                        i = self.X[data][0]
                        j = self.X[data][1]
                        e_ijk = self.TensorX[i][j][k_once] - TensorX_approximation[i][j][k_once]
                        sum2 = 0
                        for a1 in range(rr1):
                            var3 = self.U[i][a1]
                            for b1 in range(rr2):
                                temp = self.G[a1][b1][c] * self.I[j][b1] * var3
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + self.lamda * e_ijk * sum2
                if c < self.r3:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * A2[k_once][c]
                    # A2kc更新
                    self.A2[k_once][c] = A2[k_once][c] + self.theta3 * neg_gradient
                    # print("A2[{}][{}]  = {}".format(k_once, c, A2[k_once][c]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

                else:
                    sum3 = 0
                    sum4 = 0
                    for o in range(self.r):
                        sum3 = sum3 + U1[i_once][o] * A1[k_once][o]
                        sum4 = sum4 + I1[j_once][o] * A1[k_once][o]

                    explicit_gradient3 = self.lamdax * (self.Y[i_once][k_once] - sum3) * U1[i_once][c - self.r3]
                    explicit_gradient4 = self.lamday * (self.Z[j_once][k_once] - sum3) * I1[j_once][c - self.r3]
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * A1[k_once][
                        c - self.r3] + explicit_gradient3 + explicit_gradient4
                    # A1kc更新
                    self.A1[k_once][c - self.r3] = A1[k_once][c - self.r3] + self.theta3 * neg_gradient
                    # print("A1[{}][{}]  = {}".format(k_once, c - self.r3, A1[k_once][c - self.r3]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        ####################################################################################################
        # 更新核张量G,近似张量X，求重构误差
        self.U, self.I, self.A = np.concatenate((self.U1, self.U2), axis=1), \
                                 np.concatenate((self.I1, self.I2), axis=1), \
                                 np.concatenate((self.A1, self.A2), axis=1)
        factors = [self.U, self.I, self.A]
        rec_errors = []
        print("重构核张量")
        modes1 = list(range(_tucker.T.ndim(self.TensorX)))
        for index, mode in enumerate(modes1):
            self.G = _tucker.multi_mode_dot(self.TensorX, factors, modes=modes1, transpose=True)
        print("重构X张量")
        modes2 = list(range(_tucker.T.ndim(self.G)))
        for index, mode in enumerate(modes2):
            TensorX_approximation = _tucker.multi_mode_dot(self.G, factors, modes=modes2, transpose=False)
        Y_approximation = U1.dot(A1.transpose())
        Z_approximation = I1.dot(A1.transpose())


        print("计算重构误差")
        rec_X = 0
        rec_Y = 0
        rec_Z = 0
        regularizationU = _tucker.T.norm(self.TensorX, 2)
        for data in range(self.num_data):
            rec_Y = rec_Y + self.Y[self.X[data][0]][self.X[data][2]]-Y_approximation[self.X[data][0]][self.X[data][2]]
            rec_Z = rec_Z + self.Z[self.X[data][1]][self.X[data][2]]-Z_approximation[self.X[data][1]][self.X[data][2]]
            rec_X = rec_X + (self.X[data][3] - TensorX_approximation[self.X[data][0]][self.X[data][1]][
                self.X[data][2]])
        rec_error = rec_X+rec_Y+rec_Z
        self.rec_errors.append(rec_error / self.num_data)
        print("Once OK，rec={}".format(self.rec_errors))
        print("Once OK，rec={}".format(self.rec_errors), file=logfile)
        if len(rec_errors) > 2:
            print("Once OK，总误差下降幅度{}".format(self.rec_errors[-1]-self.rec_errors[-2]))
            print("Once OK，总误差下降幅度{}".format(self.rec_errors[-1] - self.rec_errors[-2]), file=logfile)

        ####################################################################################################
        # Xrec_tensor = self.TensorX-TensorX_approximation
        # print(_tucker.T.norm(Xrec_tensor, 2))
        #
        # norm_tensor = _tucker.T.norm(self.TensorX, 2)
        # rec_error = _tucker.sqrt(abs(norm_tensor ** 2 - _tucker.T.norm(TensorX_approximation, 2) ** 2)) / norm_tensor
        # norm_Y = _tucker.T.norm(self.Y, 2)
        # rec_Y = _tucker.sqrt(abs(norm_Y ** 2 - _tucker.T.norm((U1.dot(A1.transpose())), 2) ** 2)) / norm_Y
        # norm_Z = _tucker.T.norm(self.Z, 2)
        # rec_Z = _tucker.sqrt(abs(norm_Z ** 2 - _tucker.T.norm((I1.dot(A1.transpose())), 2) ** 2)) / norm_Z
        # final_error = rec_error + rec_Y + rec_Z
        # print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error))
        # print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error),
        #       file=logfile)
        # self.rec_errors.append(final_error)
        # print("Once OK，rec={}".format(self.rec_errors), file=logfile)
        # if len(rec_errors) > 2:
        #     print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2]-self.rec_errors[-1]))
        #     print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2] - self.rec_errors[-1]), file=logfile)

        return self.rec_errors, self.G, self.U, self.I, self.A, TensorX_approximation


if __name__ == "__main__":
    # 代码加载初始值#################################################################################
    # Bt = build_tensor.Build_tensor(infile="C:/Users/Syd/OneDrive/Work/EETDR/data/train.entry",
    #                                outfile="E:/PYworkspace/EETDR/result/UIA.csv")
    # X = Bt.build_tensor(tensor=False, matix=True, sprasefile=False)
    # Y = Bt.build_UA()
    # Z = Bt.build_IA()
    # X = pd.read_csv("E:/PYworkspace/EETDR/result/UIA.csv", dtype="float32", header=None, names=["user_index", "item_index", "aspect_index", "value"])
    ####################################################################################################
    # 文件加载初始值####################################################################################
    pathdir = "E:/PYworkspace/EETDR/"
    already = 0  # 指定从哪一次继续训练
    X1 = np.load(pathdir + "/data/preprodata/X1000.npy")
    Y1 = np.load(pathdir + "/data/preprodata/UA1000.npy")
    Z1 = np.load(pathdir + "/data/preprodata/IA1000.npy")
    TensorX1 = np.load(pathdir + "/data/preprodata/TensorX1000.npy")
    if already == 35:
        initU = np.load(pathdir + "/data/preprodata/initU1000.npy")
        initI = np.load(pathdir + "/data/preprodata/initI1000.npy")
        initA = np.load(pathdir + "/data/preprodata/initA1000.npy")
        core_G = np.load(pathdir + "/data/preprodata/core1000.npy")
    else:
        initU = np.load(pathdir + "/result/intermediat_result2/1000/reU" + str(already) + ".npy")
        initI = np.load(pathdir + "/result/intermediat_result2/1000/reI" + str(already) + ".npy")
        initA = np.load(pathdir + "/result/intermediat_result2/1000/reA" + str(already) + ".npy")
        core_G = np.load(pathdir + "/result/intermediat_result2/1000/reG" + str(already) + ".npy")

    ####################################################################################################
    ####################################################################################################
    factorss = [initU, initI, initA]
    TensorX_approximation_iter = tl.tucker_to_tensor(core_G, factorss)
    fine = EETDR(TensorX1, X1, Y1, Z1, core_G, initU, initI, initA)

    print("用户数{}".format(fine.num_users))
    print("产品数{}".format(fine.num_items))
    print("Aspect数{}".format(fine.num_aspects))
    print("非零数据条数{}".format(fine.num_data))
    print("显式特征向量长度{}".format(fine.r))
    print("隐式特征向量长度{} {} {}".format(fine.r1, fine.r2, fine.r3))
    print("非零数据条数{}".format(fine.num_data))
    flag = True
    count = already
    while flag:
        count += 1
        logfile = open(pathdir + "/result/log/log_fast--.txt", "a")
        print("从第0次开始使用小批量，第{}次小批量(10)".format(count), file=logfile)
        rec_errors, reG, reU, reI, reA, TensorX_approximation_iter = fine.eetdr_SGDonce(TensorX_approximation_iter)
        np.save(pathdir + "/result/intermediat_result2/1000/reG" + str(count), reG)
        np.save(pathdir + "/result/intermediat_result2/1000/reU" + str(count), reU)
        np.save(pathdir + "/result/intermediat_result2/1000/reI" + str(count), reI)
        np.save(pathdir + "/result/intermediat_result2/1000/reA" + str(count), reA)
        print(rec_errors)
        logfile.close()
        if rec_errors[-1] < 2.0 or count > 100:
            flag = False

