"""
minibatch and less reconstruction support a faster compute
"""
import tensorly as tl
import numpy as np
from tensorly.decomposition import _tucker
import build_tensor
from multiprocessing import Pool


class EETDR:

    def __init__(self, TensorX, X, Y, Z, core_G, initU, initI, initA):
        """
        :param TensorX:原始UIA稀疏张量
        :param X: TensorX,采用稀疏张量存储法：矩阵X[m][n]表示第m个非零值的n下标的值（三维张量即是n*3矩阵）最后一列X[m][-1]表示该非零值。
        :param Y:UA矩阵，用户对特征的关注度,ground_true data
        :param Z:IA矩阵，产品本身特征分布,ground_true data
        :param core_G:初始值，tucker形式分解因子中的核张量
        :param initU:初始值，tucker形式分解因子矩阵
        :param initI:初始值，tucker形式分解因子矩阵
        :param initA:初始值，tucker形式分解因子矩阵
        """
        print("初始化算法数据X, Y, Z")
        self.X = X
        self.TensorX = TensorX
        self.Y = Y
        self.Z = Z
        self.num_data = X.shape[0]
        self.num_users, self.num_items, self.num_aspects = int(Y.shape[0]), int(Z.shape[0]), int(Y.shape[1])  # 数量
        self.r = 64  # 显式因子维度
        self.r1, self.r2, self.r3 = 32, 32, 16  # 三个分解后矩阵隐式因子的数量

        print("初始化U1，I1，A1，U2，I2，A2")
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

    def get_TensorApproximateXijk(self, G, U, I, A, i, j, k, data, value):
        """
        给出tucker分解因子和坐标ijk，求出重构后张量Tijk位置元素的值
        :param G:
        :param U:
        :param I:
        :param A:
        :param i:
        :param j:
        :param k:
        :return:
        """
        TensorX_approximationXijk = 0.0
        rrU = self.r + self.r1
        rrI = self.r + self.r2
        rrA = self.r + self.r3
        for a in range(rrU):
            for b in range(rrI):
                for c in range(rrA):
                    TensorX_approximationXijk = TensorX_approximationXijk + G[a][b][c] * U[i][a] * I[j][b] * A[k][c]
        eijk = value - TensorX_approximationXijk
        return i, j, k, eijk

    def get_Eijk_store(self, G, U, I, A):
        """
        给出tucker分解因子，求出原张量T与重构张量T_approximatin,在T所有非零值处的值的差，构建一个稀疏矩阵
        :param G:
        :param U:
        :param I:
        :param A:
        :return:
        """
        p = Pool()
        Eijk = []
        res_l = []
        for data in range(self.num_data):
            i = int(self.X[data][0])
            j = int(self.X[data][1])
            k = int(self.X[data][2])
            value = int(self.X[data][3])
            result = p.apply_async(self.get_TensorApproximateXijk, (G, U, I, A, i, j, k, data, value,))
            res_l.append(result)
        p.close()
        p.join()
        for result in res_l:
            print(result.get())
            Eijk.append([result.get()])
        Eijk = np.array(Eijk).reshape((self.num_data,4))
        print('All subprocesses done. Eijk built finish')
        return Eijk

    # 获得张量分解的F范数的平方
    def get_SpraseX_Fnorm(self, Eijk):
        """
        给出原稀疏张量T在全部非零值处于重构张量T_approximatin的差的平方
        :param Eijk:
        :return:
        """
        F_error = 0
        for data in range(self.num_data):
            F_error = F_error + Eijk[data][3] ** 2
        return F_error

    def get_Sprasemat_Fnorm(self, N, N_approx):
        """
        计算原矩阵N与重构矩阵N_approx稀疏位置的差的平方
        :param N:
        :param N_approx:
        :return:
        """
        F_error = 0
        index1, index2 = N.nonzero()
        for count in range(len(index1)):
            F_error = F_error + (N[index1[count]][index2[count]] - N_approx[index1[count]][index2[count]]) ** 2
        return F_error

    def get_partial_derivative_X(self, N, X, Y, x, y):
        """
        形如（N-XY）**2的稀疏矩阵分解对Y求偏导
        :param N:
        :param X:
        :param Y:
        :param x:
        :param y:
        :return:
        """
        N_x_row = N[x, :]
        index_k = N_x_row.nonzero()
        explicit_gradient = 0
        for k in index_k:
            sum3 = 0
            for o in range(len(N_x_row)):
                sum3 = sum3 + X[x][o] * Y[k][o]
            explicit_gradient = explicit_gradient + (N[x][k] - sum3) * Y[k][y]

        return explicit_gradient

    def get_partial_derivative_Y(self, N, X, Y, x, y):
        """
        形如（N-XY）**2的稀疏矩阵分解对X求偏导
        :param N:
        :param X:
        :param Y:
        :param x:
        :param y:
        :return:
        """
        N_y_col = N[:, y]
        index_i = N_y_col.nonzero()
        explicit_gradient = 0
        for i in index_i:
            sum3 = 0
            for o in range(self.r):
                sum3 = sum3 + X[i][o] * Y[y][o]
            explicit_gradient = explicit_gradient + self.lamdax * (N[i][y] - sum3) * X[i][x]
        return explicit_gradient

    def SGD(self, G, U, I, A, U1, U2, I1, I2, A1, A2, Eijk, i_once, j_once, k_once):
        # **************************************************************************************************************
        # *******************************U2，U1的随机梯度下降更新***********************************************************

        rr1 = self.r + self.r1
        rr2 = self.r + self.r2
        rr3 = self.r + self.r3
        # 对i用户的向量每个元素进行元素级更新
        for a in range(rr1):
            sum_tensorDec = 0
            # 如下计算负梯度，公式详见论文
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
            else:
                explicit_gradient1 = self.get_partial_derivative_X(self.Y, U1, A1, i_once, a - self.r1)
                # 完全负梯度
                neg_gradient = sum_tensorDec - self.lamda2 * U1[i_once][
                    a - self.r1] + self.lamdax * explicit_gradient1
                # U1ia更新
                self.U1[i_once][a - self.r1] = U1[i_once][a - self.r1] + self.ita * neg_gradient
        # **************************************************************************************************************
        # **************************************I2，I1的随机梯度下降更新****************************************************
        # 对产品j的向量每个值进行元素级更新
        for b in range(rr2):
            sum_tensorDec = 0
            # 如下计算负梯度，公式详见论文
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
            else:
                explicit_gradient2 = self.get_partial_derivative_X(self.Z, I1, A1, j_once, b - self.r2)
                # 完全负梯度
                neg_gradient = sum_tensorDec - self.lamda2 * I1[j_once][b - self.r2] + explicit_gradient2
                # I1jb更新
                self.I1[j_once][b - self.r2] = I1[j_once][b - self.r2] + self.ita * neg_gradient
        # **************************************************************************************************************
        # **********************************A2，A1的随机梯度下降更新********************************************************
        # 对aspect k的向量每个值进行元素级更新
        for c in range(rr3):
            sum_tensorDec = 0
            # 如下计算负梯度，公式详见论文
            for data in range(self.num_data):
                if k_once == int(self.X[data][2]):
                    i = int(self.X[data][0])
                    j = int(self.X[data][1])
                    # e_ijk = self.TensorX[i][j][k_once] - TensorX_approximation[i][j][k_once]
                    e_ijk = 0
                    for count in range(self.num_data):
                        if Eijk[count][0] is i and Eijk[count][1] is j and Eijk[count][2] is k_once:
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
            else:
                explicit_gradient3 = self.get_partial_derivative_Y(self.Y, U1, A1, c - self.r3, k_once)
                explicit_gradient4 = self.get_partial_derivative_Y(self.Z, I1, A1, c - self.r3, k_once)
                # 完全负梯度
                neg_gradient = sum_tensorDec - self.lamda2 * A1[k_once][
                    c - self.r3] + explicit_gradient3 + explicit_gradient4
                # A1kc更新
                self.A1[k_once][c - self.r3] = A1[k_once][c - self.r3] + self.ita * neg_gradient
        # **************************************************************************************************************
        # **************************************************************************************************************
        self.U, self.I, self.A = np.concatenate((self.U1, self.U2), axis=1), \
                                 np.concatenate((self.I1, self.I2), axis=1), \
                                 np.concatenate((self.A1, self.A2), axis=1)

    def caculate_Final_error(self, Eijk):
        XF_error = self.get_SpraseX_Fnorm(Eijk)
        L2_U2 = _tucker.T.norm(self.U2, 2)
        L2_U1 = _tucker.T.norm(self.U1, 2)
        L2_I2 = _tucker.T.norm(self.I2, 2)
        L2_I1 = _tucker.T.norm(self.I1, 2)
        L2_A2 = _tucker.T.norm(self.A2, 2)
        L2_A1 = _tucker.T.norm(self.A1, 2)
        L2_G = _tucker.T.norm(self.G, 2)
        Y_approx = self.U1.dot(self.A1.transpose())
        Z_approx = self.I1.dot(self.A1.transpose())
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
        return self.rec_errors

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
        Eijk = self.get_Eijk_store(G, U, I, A)
        self.caculate_Final_error(Eijk)
        # **************************************************************************************************************
        print("多进程小批量次数{}".format(96))
        p = Pool()
        for i in range(300):
            i_once = i  # 随机梯度下降， 选取第i个用户更新
            j_once = i  # 随机梯度下降， 选取第i个产品更新
            k_once = i  # 随机梯度下降， 选取第i个方面更新
            p.apply_async(self.SGD, args=(G, U, I, A, U1, U2, I1, I2, A1, A2, Eijk, i_once, j_once, k_once))
        p.close()
        p.join()
        print('All subprocesses done.')
        # **************************************************************************************************************
        return self.rec_errors, self.G, self.U, self.I, self.A


if __name__ == "__main__":
    # ******************************* 文件加载初始值********************************************************************
    pathdir = "E:/PYworkspace/EETDR/"
    randominit = False
    already = 0  # 指定从哪一次继续训练
    X1 = np.load(pathdir + "/data/preprodata/X300.npy")
    Y1 = np.load(pathdir + "/data/preprodata/UA300.npy")
    Z1 = np.load(pathdir + "/data/preprodata/IA300.npy")
    TensorX1 = np.load(pathdir + "/data/preprodata/TensorX300.npy")
    if randominit:
        initU = np.random.rand(300, 96) / 10
        initI = np.random.rand(300, 96) / 10
        initA = np.random.rand(105, 80) / 10
        core_G = np.random.rand(96, 96, 80) / 10
    elif already == 0:
        initU = np.load(pathdir + "/data/preprodata/initU300_96.npy")
        initI = np.load(pathdir + "/data/preprodata/initI300_96.npy")
        initA = np.load(pathdir + "/data/preprodata/initA300_96.npy")
        core_G = np.load(pathdir + "/data/preprodata/core300_96.npy")
    else:
        initU = np.load(pathdir + "/result/300_RE_96/reU" + str(already) + ".npy")
        initI = np.load(pathdir + "/result/300_RE_96/reI" + str(already) + ".npy")
        initA = np.load(pathdir + "/result/300_RE_96/reA" + str(already) + ".npy")
        core_G = np.load(pathdir + "/result/300_RE_96/reG" + str(already) + ".npy")

    ####################################################################################################################
    ####################################################################################################################

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
        logfile = open(pathdir + "/result/log/log_fast_300_RE_64.txt", "a")
        rec_errors, reG, reU, reI, reA = fine.eetdr_SGDonce()
        np.save(pathdir + "/result/300_RE_96/reG" + str(count), reG)
        np.save(pathdir + "/result/300_RE_96/reU" + str(count), reU)
        np.save(pathdir + "/result/300_RE_96/reI" + str(count), reI)
        np.save(pathdir + "/result/300_RE_96/reA" + str(count), reA)
        print(rec_errors)
        print(rec_errors, file=logfile)
        if len(rec_errors) > 2:
            print("RiseRate = {}".format(rec_errors[-1] - rec_errors[-2]))
            print("RiseRate = {}".format(rec_errors[-1] - rec_errors[-2]), file=logfile)
            if rec_errors[-2] - rec_errors[-1] < 0.1 or count > 5000:
                flag = False
        logfile.close()
