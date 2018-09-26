# _*_ coding:utf-8 _*_
__AUTHOR__ = "syd"
# DATA:2018/9/13
# PROJECT:Pyworkplace
"""
尝试使用pytorch 加速
"""
import random
import pandas as pd
import tensorly as tl
import numpy as np
from tensorly.decomposition import _tucker
from tensorly.decomposition._tucker import multi_mode_dot
import build_tensor
import time
import torch as tr
from torch.autograd import Variable


class EETDR:

    def __init__(self, TensorX, X, Y, Z, core_G, initU, initI, initA):
        print("初始化算法数据X, Y, Z")
        # 原始UIA构成的张量，ground_true data
        # 采用稀疏张量存储法：矩阵X[m][n]表示第m个非零值的n下标的值（三维张量即是n*3矩阵）。
        # 最后一列X[m][-1]表示该非零值
        self.X = tr.from_numpy(X).cuda()
        self.TensorX = tr.from_numpy(TensorX).cuda()
        # UA矩阵，用户对特征的关注度,ground_true data
        self.Y = tr.from_numpy(Y).cuda()
        # IA矩阵，产品本身特征分布,ground_true data
        self.Z = tr.from_numpy(Z).cuda()
        self.num_data = X.shape[0]
        self.num_users, self.num_items, self.num_aspects = int(Y.shape[0]), int(Z.shape[0]), int(Y.shape[1])  # 数量
        self.r = 32  # 显式因子维度
        self.r1, self.r2, self.r3 = 32, 32, 16  # 三个分解后矩阵隐式因子的数量
        print("初始化U1，I1，A1，U2，I2，A2")
        # 初始值U，I，A矩阵由tucker分解一次得到
        self.G = tr.from_numpy(core_G).cuda()
        self.U = tr.from_numpy(initU).cuda()
        self.I = tr.from_numpy(initI).cuda()
        self.A = tr.from_numpy(initA).cuda()
        self.U2 = tr.from_numpy(initU[:, :self.r1]).cuda()
        self.I2 = tr.from_numpy(initI[:, :self.r2]).cuda()
        self.A2 = tr.from_numpy(initA[:, :self.r3]).cuda()
        self.U1 = tr.from_numpy(initU[:, self.r1:]).cuda()
        self.I1 = tr.from_numpy(initI[:, self.r2:]).cuda()
        self.A1 = tr.from_numpy(initA[:, self.r3:]).cuda()

        self.ita = 0.1  # 所有变量的学习率 yita
        self.lamdax = 0.3  # Y - U1 * A1
        self.lamday = 0.3  # Z - I1 * A1
        self.lamda1 = 0.1  # U1，I1，A1的正则化参数
        self.lamda2 = 0.1  # U2，I2，A2的正则化参数
        self.lamda3 = 0.1  # 核张量G的正则化参数
        print("初始化参数，学习率ita = {},lamdax={}，lamday={}，lamda1={}，lamda2={}，lamda3={}".format(self.ita, self.lamdax,
                                                                                           self.lamday, self.lamda1,
                                                                                           self.lamda2, self.lamda3))

        self.rec_errors = []
        return

    def get_TensorApproximateXijk(self, G, U, I, A, i, j, k):
        TensorX_approximationXijk = 0
        rrU = self.r + self.r1
        rrI = self.r + self.r2
        rrA = self.r + self.r3
        for a in range(rrU):
            for b in range(rrI):
                for c in range(rrA):
                    TensorX_approximationXijk = G[a][b][c] * U[i][a] * I[j][b] * A[k][c]
        return TensorX_approximationXijk

    def get_Eijk_store(self, G, U, I, A):
        Eijk = np.ndarray((self.num_data, 4))
        for data in range(self.num_data):
            i = int(self.X[data][0])
            j = int(self.X[data][1])
            k = int(self.X[data][2])
            Eijk[data][0] = i
            Eijk[data][1] = j
            Eijk[data][2] = k
            Eijk[data][3] = self.X[data][3] - self.get_TensorApproximateXijk(G, U, I, A, i, j, k)
        return Eijk

    # 获得张量分解的F范数的平方
    def get_SpraseX_Fnorm(self, Eijk):
        F_error = 0
        for data in range(self.num_data):
            F_error = F_error + Eijk[data][3] ** 2
        return F_error

    def get_Sprasemat_Fnorm(self, N, N_approx):
        F_error = 0
        index1, index2 = N.nonzero()
        for count in range(len(index1)):
            F_error = F_error + (N[index1[count]][index2[count]] - N_approx[index1[count]][index2[count]]) ** 2
        return F_error

    def eetdr_SGDonce(self):
        U = self.U
        I = self.I
        A = self.A
        U1 = self.U1
        U2 = self.U2
        I1 = self.I1
        I2 = self.I2
        A1 = self.A1
        A2 = self.A2
        G = self.G
        #########总误差
        Eijk = self.get_Eijk_store(G, U, I, A)
        XF_error = self.get_SpraseX_Fnorm(Eijk)
        L2_U2 = _tucker.T.norm(self.U2, 2)
        L2_U1 = _tucker.T.norm(self.U1, 2)
        L2_I2 = _tucker.T.norm(self.I2, 2)
        L2_I1 = _tucker.T.norm(self.I1, 2)
        L2_A2 = _tucker.T.norm(self.A2, 2)
        L2_A1 = _tucker.T.norm(self.A1, 2)
        L2_G = _tucker.T.norm(self.G, 2)
        Y_approx = U1.dot(A1.transpose())
        Z_approx = I1.dot(A1.transpose())
        Y_error = self.get_Sprasemat_Fnorm(self.Y, Y_approx)
        Z_error = self.get_Sprasemat_Fnorm(self.Z, Z_approx)
        final_error = XF_error + self.lamdax * Y_error + self.lamday * Z_error + self.lamda1 * (
                L2_U1 + L2_A1 + L2_I1) + self.lamda2 * (L2_U2 + L2_A2 + L2_I2) + self.lamda3 * L2_G
        print("XF_error={}", XF_error)
        print("Y_error={}", Y_error)
        print("Z_error={}", Z_error)
        print("L2_G={}", L2_G)
        print("L2_U1 + L2_A1 + L2_I1={}", L2_U1 + L2_A1 + L2_I1)
        print("L2_U2 + L2_A2 + L2_I2={}", L2_U2 + L2_A2 + L2_I2)

        self.rec_errors.append(final_error)
        print("final_error={}".format(final_error))
        print("final_error={}".format(final_error), file=logfile)
        ########################################################
        print("稀疏矩阵小批量(50)梯度下降")
        for minibatch in range(100):
            print("小批量次数{}".format(minibatch))
            ##############################################################################################
            # U2的随机梯度下降更新########################################################################
            seed = np.random.randint(0, self.num_data)
            i_once = int(self.X[seed][0])  # 随机梯度下降， 随机选取第i个用户更新
            j_once = int(self.X[seed][1])  # 随机梯度下降， 随机选取第i个用户更新
            k_once = int(self.X[seed][2])  # 随机梯度下降， 随机选取第i个用户更新
            print("updata[{}][{}][{}]".format(i_once, j_once, k_once))
            rr1 = self.r + self.r1
            rr2 = self.r + self.r2
            rr3 = self.r + self.r3
            Ur = self.r1 + self.r
            # 对i用户的向量每个元素进行元素级更新
            for a in range(Ur):
                # print("updata U[{}][{}]".format(i_once, a))
                # print("U[{}][{}]  = {}".format(i_once, a, U[i_once][a]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if i_once == int(self.X[data][0]):
                        j = int(self.X[data][1])
                        k = int(self.X[data][2])
                        e_ijk = 0
                        # e_ijk = self.TensorX[i_once][j][k] - TensorX_approximation[i_once][j][k]
                        for count in range(self.num_data):
                            if Eijk[count][0] is i_once and Eijk[count][1] is j and Eijk[count][2] is k:
                                e_ijk = Eijk[count][3]

                        sum2 = 0
                        for b1 in range(rr2):
                            var3 = I[j][b1]
                            for c1 in range(rr3):
                                temp = G[a][b1][c1] * var3 * A[k][c1]
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + e_ijk * sum2
                if a < self.r1:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * U2[i_once][a]
                    # U2ia更新
                    self.U2[i_once][a] = U2[i_once][a] + self.ita * neg_gradient

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
                    self.U1[i_once][a - self.r1] = U1[i_once][a - self.r1] + self.ita * neg_gradient
                    # print("U1[{}][{}]  = {}".format(i_once, a - self.r1, U1[i_once][a - self.r1]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

            ##############################################################################################
            # I2的随机梯度下降更新########################################################################
            Ir = self.r2 + self.r
            # 对产品j的向量每个值进行元素级更新
            for b in range(Ir):
                # print("updata I[{}][{}]".format(j_once, b))
                # print("I[{}][{}]  = {}".format(j_once, b, I[j_once][b]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if j_once == int(self.X[data][1]):
                        i = int(self.X[data][0])
                        k = int(self.X[data][2])
                        e_ijk = 0
                        for count in range(self.num_data):
                            if Eijk[count][0] is i and Eijk[count][1] is j_once and Eijk[count][2] is k:
                                e_ijk = Eijk[count][3]
                        sum2 = 0
                        for a1 in range(rr1):
                            var3 = U[i][a1]
                            for c1 in range(rr3):
                                temp = G[a1][b][c1] * var3 * A[k][c1]
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + e_ijk * sum2
                if b < self.r2:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * I2[j_once][b]
                    # I2jb更新
                    self.I2[j_once][b] = I2[j_once][b] + self.ita * neg_gradient
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
                    self.I1[j_once][b - self.r2] = I1[j_once][b - self.r2] + self.ita * neg_gradient
                    # print("I1[{}][{}]  = {}".format(j_once, b - self.r2, I1[j_once][b - self.r2]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

            ##############################################################################################
            # A2的随机梯度下降更新########################################################################
            Ar = self.r3 + self.r
            # 对aspect k的向量每个值进行元素级更新
            for c in range(Ar):
                # print("updata A[{}][{}]".format(k_once, c))
                # print("A[{}][{}]  = {}".format(k_once, c, A[k_once][c]), file=logfile)
                # 稀疏张量分解得到的负梯度
                sum_tensorDec = 0
                # 如下计算负梯度，公式详见论文
                # 稀疏张量的取下标。
                for data in range(self.num_data):
                    if k_once == int(self.X[data][2]):
                        i = int(self.X[data][0])
                        j = int(self.X[data][1])
                        # e_ijk = self.TensorX[i][j][k_once] - TensorX_approximation[i][j][k_once]
                        e_ijk = 0
                        for count in range(self.num_data):
                            if Eijk[count][0] is i and Eijk[count][1] is j_once and Eijk[count][2] is k:
                                e_ijk = Eijk[count][3]
                        sum2 = 0
                        for a1 in range(rr1):
                            var3 = U[i][a1]
                            for b1 in range(rr2):
                                temp = G[a1][b1][c] * I[j][b1] * var3
                                sum2 += temp
                        sum_tensorDec = sum_tensorDec + e_ijk * sum2
                if c < self.r3:
                    # 完全负梯度
                    neg_gradient = sum_tensorDec - self.lamda2 * A2[k_once][c]
                    # A2kc更新
                    self.A2[k_once][c] = A2[k_once][c] + self.ita * neg_gradient
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
                    self.A1[k_once][c - self.r3] = A1[k_once][c - self.r3] + self.ita * neg_gradient
                    # print("A1[{}][{}]  = {}".format(k_once, c - self.r3, A1[k_once][c - self.r3]), file=logfile)
                    # print("↑完全负梯度{}".format(neg_gradient), file=logfile)

        ####################################################################################################
        # 更新核张量G,近似张量X，求重构误差
        self.U, self.I, self.A = np.concatenate((self.U1, self.U2), axis=1), \
                                 np.concatenate((self.I1, self.I2), axis=1), \
                                 np.concatenate((self.A1, self.A2), axis=1)
        # factors = [self.U, self.I, self.A]
        # print("重构核张量")
        # modes1 = list(range(_tucker.T.ndim(self.TensorX)))
        # self.G = _tucker.multi_mode_dot(self.TensorX, factors, modes=modes1, transpose=True)
        # print("重构X张量")
        # modes2 = list(range(_tucker.T.ndim(self.G)))
        # TensorX_approximation = _tucker.multi_mode_dot(self.G, factors, modes=modes2, transpose=False)
        # print("计算重构误差")
        ####################################################################################################
        # L2_U2 = _tucker.T.norm(self.U2, 2)
        # L2_U1 = _tucker.T.norm(self.U1, 2)
        # L2_I2 = _tucker.T.norm(self.I2, 2)
        # L2_I1 = _tucker.T.norm(self.I1, 2)
        # L2_A2 = _tucker.T.norm(self.A2, 2)
        # L2_A1 = _tucker.T.norm(self.A1, 2)
        # L2_G = _tucker.T.norm(self.G, 2)
        # norm_tensor = _tucker.T.norm(self.TensorX, 2)
        # rec_error = _tucker.sqrt(abs(norm_tensor ** 2 - _tucker.T.norm(TensorX_approximation, 2) ** 2)) / norm_tensor
        # norm_Y = _tucker.T.norm(self.Y, 2)
        # rec_Y = _tucker.sqrt(abs(norm_Y ** 2 - _tucker.T.norm((U1.dot(A1.transpose())), 2) ** 2)) / norm_Y
        # norm_Z = _tucker.T.norm(self.Z, 2)
        # rec_Z = _tucker.sqrt(abs(norm_Z ** 2 - _tucker.T.norm((I1.dot(A1.transpose())), 2) ** 2)) / norm_Z
        # E = self.lamda1 * (L2_U1 + L2_A1 + L2_I1) + self.lamda2 * (L2_U2 + L2_A2 + L2_I2) + self.lamda3 * L2_G
        # final_error = rec_error + self.lamdax * rec_Y + self.lamday * rec_Z + self.lamda1 * (
        #         L2_U1 + L2_A1 + L2_I1) + self.lamda2 * (L2_U2 + L2_A2 + L2_I2) + self.lamda3 * L2_G
        # print(E)
        # print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error))
        # print(" finalY+Z={}".format(rec_Y + rec_Z),
        #       file=logfile)
        # print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error),
        #       file=logfile)
        # self.rec_errors.append(final_error)
        # print("Once OK，rec={}".format(self.rec_errors), file=logfile)
        # if len(self.rec_errors) > 2:
        #     print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2] - self.rec_errors[-1]))
        #     print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2] - self.rec_errors[-1]), file=logfile)

        return self.rec_errors, self.G, self.U, self.I, self.A


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
    randominit = True
    already = 5  # 指定从哪一次继续训练
    X1 = np.load(pathdir + "/data/preprodata/X300.npy")
    Y1 = np.load(pathdir + "/data/preprodata/UA300.npy")
    Z1 = np.load(pathdir + "/data/preprodata/IA300.npy")
    TensorX1 = np.load(pathdir + "/data/preprodata/TensorX300.npy")
    if randominit:
        initU = np.random.rand(300, 64) / 10
        initI = np.random.rand(300, 64) / 10
        initA = np.random.rand(105, 48) / 10
        core_G = np.random.rand(64, 64, 48) / 10
    elif already == 0:
        initU = np.load(pathdir + "/data/preprodata/initU300.npy")
        initI = np.load(pathdir + "/data/preprodata/initI300.npy")
        initA = np.load(pathdir + "/data/preprodata/initA300.npy")
        core_G = np.load(pathdir + "/data/preprodata/core300.npy")
    else:
        initU = np.load(pathdir + "/result/intermediat_result2/300_0.1/reU" + str(already) + ".npy")
        initI = np.load(pathdir + "/result/intermediat_result2/300_0.1/reI" + str(already) + ".npy")
        initA = np.load(pathdir + "/result/intermediat_result2/300_0.1/reA" + str(already) + ".npy")
        core_G = np.load(pathdir + "/result/intermediat_result2/300_0.1/reG" + str(already) + ".npy")

    ####################################################################################################
    ####################################################################################################

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
        logfile = open(pathdir + "/result/log/log_fast_300Final_0.1.txt", "a")
        print("从第0次开始使用小批量，第{}次小批量(50)".format(count), file=logfile)
        rec_errors, reG, reU, reI, reA = fine.eetdr_SGDonce()
        np.save(pathdir + "/result/intermediat_result2/300_0.1/reG" + str(count), reG)
        np.save(pathdir + "/result/intermediat_result2/300_0.1/reU" + str(count), reU)
        np.save(pathdir + "/result/intermediat_result2/300_0.1/reI" + str(count), reI)
        np.save(pathdir + "/result/intermediat_result2/300_0.1/reA" + str(count), reA)
        print(rec_errors)
        print(rec_errors, file=logfile)
        if len(rec_errors) > 2:
            print("RiseRate = {}".format(rec_errors[-1] - rec_errors[-2]))
            print("RiseRate = {}".format(rec_errors[-1] - rec_errors[-2]), file=logfile)
            if rec_errors[-2] - rec_errors[-1] < 1 or count > 500:
                flag = False
        logfile.close()
