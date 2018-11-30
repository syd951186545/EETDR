import autograd.numpy as np
from autograd import multigrad
import time
import random


def construct_IF_W_vec(I, F, W, if_pair):
    A = np.hstack((I[if_pair[0]], F[if_pair[1]]))
    # 某个产品的某个特征的词向量
    if_W = np.einsum("a,ma->m", A, W)
    return if_W


def construct_IF_U_row(U, I, F, u_id, i_id, item_f_pair_list2):
    # 某用户对某产品各个特征F的评分
    U_if_all = {}
    for a in range(len(item_f_pair_list2)):
        A = np.hstack((I[i_id], F[item_f_pair_list2[a]]))
        U_if_all[item_f_pair_list2[a]] = np.einsum("a,a->", A, U[u_id])
    return U_if_all


def cost_abs_sparse_BPR_SGD(U, I, F, W, sps_tensor_useritemf, sps_tensor_ifw,
                            element_list_useritemf, element_list_ifw, element_list_ifw_2,
                            sps_overall_rating, element_list_overall_rating,
                            ):
    error_U_IF = 0
    error_W_IF = 0
    error_Rating = 0
    bpr_word_error = 0
    W_num = 1019
    element_num_iter = 300
    lmd_reg = 0.05
    lmd_bpr = 100

    # loss of Rating
    for kkk in range(element_num_iter):
        [key] = random.sample(element_list_overall_rating, 1)
        index = key[1:-1].split(',')
        for i in range(2):
            index[i] = int(index[i])
        IFr = np.hstack((I[index[1]], F[104]))
        value = np.einsum("a,a->", U[index[0]], IFr)
        error_Rating += (value - sps_overall_rating[key]) ** 2
    error_Rating = error_Rating / element_num_iter
    # loss of item/feature recommendation task
    # loss of UI_F
    for kkk in range(element_num_iter):
        # 训练的是uif有实值的对，i没有的f不会被训练，实际上分解的矩阵是不包含if（其中f不属于i）
        [key] = random.sample(element_list_useritemf, 1)
        index = key[1:-1].split(',')
        for i in range(3):
            index[i] = int(index[i])
        A = np.hstack((I[index[1]], F[index[2]]))
        value = np.einsum("a,a->", A, U[index[0]])
        error_U_IF += (value - sps_tensor_useritemf[key]) ** 2
    # 小批量300样本随机梯度下降，误差/300取平均
    error_U_IF = error_U_IF / element_num_iter

    # loss of W_IF
    for kkk in range(element_num_iter):
        # 实际存在的ifw对的值的损失
        [key] = random.sample(element_list_ifw, 1)
        index = key[1:-1].split(',')
        for i in range(3):
            index[i] = int(index[i])
        word_i = index[2]
        A = np.hstack((I[index[0]], F[index[1]]))
        value_i = np.einsum("a,a->", A, W[word_i])
        error_W_IF += (value_i - sps_tensor_ifw[key]) ** 2
    error_W_IF = error_W_IF / element_num_iter
    # BPR_W
    for kkk in range(100):
        [key] = random.sample(element_list_ifw_2, 1)
        index = key[1:-1].split(',')
        for i in range(3):
            index[i] = int(index[i])
        word_i = index[2]
        word_j = int(W_num * random.random())
        # 原始ifw集中的大小要保持
        if str([index[0], index[1], word_j]) in sps_tensor_ifw.keys():
            if sps_tensor_ifw[key] < sps_tensor_ifw[str([index[0], index[1], word_j])]:
                value_i = np.einsum("a,a->", A, W[word_i])
                value_j = np.einsum("a,a->", A, W[word_j])
                bpr_word_error += (0 - np.log(1 / (1 + np.exp(value_i - value_j))))
            # 随机到了同一个词
            if sps_tensor_ifw[key] == sps_tensor_ifw[str([index[0], index[1], word_j])]:
                bpr_word_error += 0
        else:
            value_i = np.einsum("a,a->", A, W[word_i])
            value_j = np.einsum("a,a->", A, W[word_j])
            bpr_word_error += (0 - np.log(1 / (1 + np.exp(value_j - value_i))))
    bpr_word_error = bpr_word_error / 100

    # loss of bpr word

    error_reg = 0
    error = U.flatten()
    error_reg += np.sqrt((error ** 2).mean())
    error = I.flatten()
    error_reg += np.sqrt((error ** 2).mean())
    error = F.flatten()
    error_reg += np.sqrt((error ** 2).mean())
    error = W.flatten()
    error_reg += np.sqrt((error ** 2).mean())

    print('Least error_Rating:')
    print(error_Rating)
    print('Least error_U_IF:')
    print(error_U_IF)
    print('Least error_W_IF:')
    print(error_W_IF)
    print('Least bpr_word_error:')
    print(bpr_word_error)
    print("Total lost:")
    print(error_U_IF + error_W_IF + error_Rating + bpr_word_error)
    # return error1
    return error_U_IF + 0.05*error_W_IF + error_Rating + lmd_bpr * bpr_word_error + lmd_reg * error_reg


def learn_HAT_SGD_adagrad(sps_tensor_useritemf, sps_tensor_ifw, sps_overall_rating,
                          U_dim, I_dim, F_dim,
                          W_dim, U_num, I_num,
                          F_num_1more, W_num,
                          num_iter=100000, lr=0.1,
                          dis=False, cost_function='abs', random_seed=0,
                          eps=1e-8):
    F_num = F_num_1more - 1
    np.random.seed(random_seed)
    cost = cost_abs_sparse_BPR_SGD
    element_list_useritemf = list(sps_tensor_useritemf)
    element_list_ifw = list(sps_tensor_ifw)
    element_list_ifw_2 = []
    for item in sps_tensor_ifw.items():
        if item[1] > 3.93:
            element_list_ifw_2.append(item[0])
    element_list_overall_rating = list(sps_overall_rating)

    params = {}
    params['M'], params['N'], params['F'], params['W'] = (U_num, I_num, F_num, W_num)
    '''
	params['a'] = U0_dim
	params['b'] = U1_dim
	params['c'] = U2_dim
	params['d'] = I0_dim
	params['e'] = I1_dim
	params['f'] = I2_dim
	params['g'] = F_dim
	params['h'] = W_dim
	'''
    print("users:" + str(params['M']))
    print("items:" + str(params['N']))
    print("features:" + str(params['F']))
    print("words:" + str(params['W']))

    U_dim_initial = (U_num, U_dim)
    I_dim_initial = (I_num, I_dim)
    F_dim_initial = (F_num_1more, F_dim)
    W_dim_initial = (W_num, W_dim)

    U = np.random.rand(*U_dim_initial)
    I = np.random.rand(*I_dim_initial)
    F = np.random.rand(*F_dim_initial)
    W = np.random.rand(*W_dim_initial)

    sum_square_gradients_U = np.zeros_like(U)
    sum_square_gradients_I = np.zeros_like(I)
    sum_square_gradients_F = np.zeros_like(F)
    sum_square_gradients_W = np.zeros_like(W)

    mg = multigrad(cost, argnums=[0, 1, 2, 3])

    # SGD procedure
    for i in range(num_iter):
        starttime = time.time()
        print(i + 1)
        # print('?')
        del_u, del_i, del_f, del_w = mg(U, I, F, W,
                                        sps_tensor_useritemf, sps_tensor_ifw,
                                        element_list_useritemf, element_list_ifw, element_list_ifw_2,
                                        sps_overall_rating, element_list_overall_rating)

        # eps+del_g**2

        sum_square_gradients_U += eps + np.square(del_u)
        sum_square_gradients_I += eps + np.square(del_i)
        sum_square_gradients_F += eps + np.square(del_f)
        sum_square_gradients_W += eps + np.square(del_w)

        # np.divide()对位除法只保留整数部分，np.sqrt()各元素平方根 lr=0.1，# 0.1/((eps+del_g**2)**1/2)

        lr_u = np.divide(lr, np.sqrt(sum_square_gradients_U))

        lr_i = np.divide(lr, np.sqrt(sum_square_gradients_I))

        lr_f = np.divide(lr, np.sqrt(sum_square_gradients_F))

        lr_w = np.divide(lr, np.sqrt(sum_square_gradients_W))

        # 梯度下降 G1=G1 - 0.1/((eps+del_g**2)**1/2) * del_g

        U -= lr_u * del_u
        I -= lr_i * del_i
        F -= lr_f * del_f
        W -= lr_w * del_w

        # Projection to non-negative space

        U[U < 0] = 0
        I[I < 0] = 0
        F[F < 0] = 0
        W[W < 0] = 0

        nowtime = time.time()
        timeleft = (nowtime - starttime) * (num_iter - i - 1)

        if timeleft / 60 > 60:
            print('time left: ' + str(int(timeleft / 3600)) + ' hr ' + str(int(timeleft / 60 % 60)) + ' min ' + str(
                int(timeleft % 60)) + ' s')
        else:
            print("time left: " + str(int(timeleft / 60)) + ' min ' + str(int(timeleft % 60)) + ' s')

    return U, I, F, W
