import numpy as np
import metrics
import tensorly
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_groundRating_mat():
    path = "E:/PYworkspace/EETDR/data/testdata"
    groundRating_mat = np.load(path + "/1000test_mat.npy")
    return groundRating_mat


def get_re_tensor_element():
    path = "E:/PYworkspace/EETDR/result/intermediat_result2/1000_2"
    core = np.load(path + "/reG14.npy")
    U = np.load(path + "/reU14.npy")
    I = np.load(path + "/reI14.npy")
    A = np.load(path + "/reA14.npy")

    return core, U, I, A


def reconstruct_tensor():
    core, U, I, A = get_re_tensor_element()
    factors = [U, I, A]
    TensorX_approximation = tensorly.tucker_to_tensor(core, factors, )
    return TensorX_approximation


def predict_rating(TensorX_approximation, top_K, userid, itemid, alpha):
    """
    任意一个用户对任意一个产品的推荐预测分
    :param TensorX_approximation:重构张量
    :param top_K:最大的前top_K项aspect预测分
    :param userid:预测用户号
    :param itemid:预测商品号
    :param alpha：aspect和rating对于预测分值的比重
    :param alpha：aspect和rating对于预测分值的比重
    :return:用户对各个aspect的预测分，推荐预测分
    """

    rating_score = 0
    aspect_rating_score_list = TensorX_approximation[userid][itemid][:-2]
    rating_s = (1 - alpha) * TensorX_approximation[userid][itemid][-1]
    np.ndarray.sort(aspect_rating_score_list)
    for k in range(top_K):
        rating_score = rating_score + alpha * aspect_rating_score_list[-k]
    rating_score = rating_score / top_K + rating_s
    return aspect_rating_score_list, rating_score


def predict_score_rank(TensorX_approximation, top_K_aspects, num_users, num_items, alpha=0.3):
    """
    得到预测分所构成的矩阵
    :param TensorX_approximation:reconstruction tensor
    :param top_K_aspects:max top_K scores of aspect
    :param num_users:numbers of users
    :param num_items:number of items
    :return: approximate Rating mat
    """
    approximate_Rating_mat = np.zeros(shape=(num_users, num_items))
    for user in range(num_users):
        for item in range(num_items):
            _, item_score = predict_rating(TensorX_approximation, top_K_aspects, user, item, alpha)
            approximate_Rating_mat[user][item] = item_score
    return approximate_Rating_mat


def ndcg():
    return


if __name__ == "__main__":
    TensorX_approximation = reconstruct_tensor()
    groundRating_mat = get_groundRating_mat()
    approximate_Rating_mat = predict_score_rank(TensorX_approximation, 50, 1000, 1000, alpha=0.3)

    print(TensorX_approximation[1][0][104])
    print(TensorX_approximation[2][0][104])
    print(approximate_Rating_mat[5][0])
    print(approximate_Rating_mat[20][1])
    # fig1 = plt.figure()
    # fig2 = plt.figure()
    # ax = Axes3D(fig1)
    # ax2 = Axes3D(fig2)
    # t = np.arange(1000)
    # X, Y = np.meshgrid(t, t)
    # ax.plot_wireframe(X, Y, approximate_Rating_mat, rstride=1, cstride=1, cmap='rainbow')
    # ax2.plot_surface(X, Y, groundRating_mat, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
