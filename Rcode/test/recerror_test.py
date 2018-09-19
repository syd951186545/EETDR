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
import EETDR.Rcode.build_tensor
import time


class EETDR:

    def __init__(self, TensorX, X, Y, Z, core_G, initU, initI, initA, logfile):
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
        self.rec_errors = []
        return

    def eetdr_SGDonce(self, TensorX_approximation):
###############################################################################################
        # 更新核张量G,近似张量X，求重构误差
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
        print("计算重构误差")
        # rec_error = 0
        # for data in range(self.num_data):
        #     rec_error = rec_error + self.X[data][3] - TensorX_approximation[self.X[data][0]][self.X[data][1]][
        #         self.X[data][2]]
        # self.rec_errors.append(rec_error / self.num_data)
        # print("Once OK，rec={}".format(self.rec_errors))
        # print("Once OK，rec={}".format(self.rec_errors), file=logfile)
        ####################################################################################################
        norm_tensor = _tucker.T.norm(self.TensorX, 2)
        rec_error = _tucker.sqrt(abs(norm_tensor ** 2 - _tucker.T.norm(TensorX_approximation, 2) ** 2)) / norm_tensor
        norm_Y = _tucker.T.norm(self.Y, 2)
        rec_Y = _tucker.sqrt(abs(norm_Y ** 2 - _tucker.T.norm((self.U1.dot(self.A1.transpose())), 2) ** 2)) / norm_Y
        norm_Z = _tucker.T.norm(self.Z, 2)
        rec_Z = _tucker.sqrt(abs(norm_Z ** 2 - _tucker.T.norm((self.I1.dot(self.A1.transpose())), 2) ** 2)) / norm_Z
        final_error = rec_error + rec_Y + rec_Z
        print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error))
        print("rec_error={} rec_Y={} rec_Z={} final={}".format(rec_error, rec_Y, rec_Z, final_error),
              file=logfile)
        self.rec_errors.append(final_error)
        print("Once OK，rec={}".format(self.rec_errors), file=logfile)
        if len(rec_errors) > 2:
            print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2]-self.rec_errors[-1]))
            print("Once OK，总误差下降幅度{}".format(self.rec_errors[-2] - self.rec_errors[-1]), file=logfile)

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
    already = 17  # 指定从哪一次继续训练
    X1 = np.load(pathdir + "/data/preprodata/X3000.npy")
    Y1 = np.load(pathdir + "/data/preprodata/UA3000.npy")
    Z1 = np.load(pathdir + "/data/preprodata/IA3000.npy")
    TensorX1 = np.load(pathdir + "/data/preprodata/TensorX.npy")
    if already == 0:
        initU = np.load(pathdir + "/data/preprodata/initU.npy")
        initI = np.load(pathdir + "/data/preprodata/initI.npy")
        initA = np.load(pathdir + "/data/preprodata/initA.npy")
        core_G = np.load(pathdir + "/data/preprodata/core.npy")
    else:
        initU = np.load(pathdir + "/result/intermediat_result2/reU" + str(already) + ".npy")
        initI = np.load(pathdir + "/result/intermediat_result2/reI" + str(already) + ".npy")
        initA = np.load(pathdir + "/result/intermediat_result2/reA" + str(already) + ".npy")
        core_G = np.load(pathdir + "/result/intermediat_result2/reG" + str(already) + ".npy")
    logfile = open(pathdir + "/result/log/log_fast++.txt", "a")
    ####################################################################################################
    ####################################################################################################
    factorss = [initU, initI, initA]
    TensorX_approximation_iter = tl.tucker_to_tensor(core_G, factorss)
    fine = EETDR(TensorX1, X1, Y1, Z1, core_G, initU, initI, initA, logfile)
    logfile.close()
    print("用户数{}".format(fine.num_users))
    print("产品数{}".format(fine.num_items))
    print("Aspect数{}".format(fine.num_aspects))
    print("非零数据条数{}".format(fine.num_data))
    print("显式特征向量长度{}".format(fine.r))
    print("隐式特征向量长度{} {} {}".format(fine.r1, fine.r2, fine.r3))
    print("非零数据条数{}".format(fine.num_data))
    flag = True
    count = already-57
    while flag:
        count += 1
        logfile = open(pathdir + "/result/log/log_fast++.txt", "a")
        print("从第54次开始使用小批量，第{}次小批量(10)".format(count), file=logfile)
        rec_errors, reG, reU, reI, reA, TensorX_approximation_iter = fine.eetdr_SGDonce(TensorX_approximation_iter)
        print(rec_errors)
        logfile.close()
        if rec_errors[-1] < 2.0:
            flag = False

