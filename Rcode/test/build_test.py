import numpy as np
import metrics
import tensorly
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_testRating_mat():
    path = "E:/PYworkspace/EETDR/data/testdata"
    groundRating_mat = np.load(path + "/300test_mat.npy")
    return groundRating_mat


def get_re_tensor_element(num):
    path = "E:/PYworkspace/EETDR/result/intermediat_result2/300_0.1"
    core = np.load(path + "/reG" + num + ".npy")
    U = np.load(path + "/reU" + num + ".npy")
    I = np.load(path + "/reI" + num + ".npy")
    A = np.load(path + "/reA" + num + ".npy")

    return core, U, I, A


def reconstruct_tensor(num):
    core, U, I, A = get_re_tensor_element(num)
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


def get_DCG(groundRating_mat, approximate_Rating_mat, recom_topK):
    numusers, numitems = groundRating_mat.shape
    DCG_rank_mat = np.ones((numusers, recom_topK))
    for userid in range(numusers):
        user_approx_rank = approximate_Rating_mat[userid, :]
        user_true_rank = groundRating_mat[userid, :]
        item_indx = np.argpartition(user_approx_rank,  range(recom_topK))
        for numid in range(recom_topK):
            if user_true_rank[item_indx[numid]] == 0.0:
                # DCG_rank_mat[userid][numid] = 1  # fire = 2**1-1
                continue
            else:
                rating_rec = abs(user_true_rank[item_indx[numid]] - user_approx_rank[item_indx[numid]])
                if rating_rec <= 0.5:
                    DCG_rank_mat[userid][numid] = 15  # perfect =2**4-1
                if 0.5 < rating_rec <= 1:
                    DCG_rank_mat[userid][numid] = 7  # excellent =2**3-1
                if 1 < rating_rec <= 2:
                    DCG_rank_mat[userid][numid] = 3  # good =2**2-1
                if rating_rec > 2:
                    DCG_rank_mat[userid][numid] = 0  # bad =2**0-1
    # print(DCG_rank_mat)
    # print("********************")
    return DCG_rank_mat

def get_NDCG(DCG_rank_mat):
    num_users,num_recom_topK=DCG_rank_mat.shape
    avg_ndcg_score = 0.0
    for user in range(num_users):
        user_dcg_rank = DCG_rank_mat[user, :]
        ndcg_score = metrics.get_ndcg(user_dcg_rank,num_recom_topK)
        avg_ndcg_score += ndcg_score/num_users
        print(ndcg_score)
    return avg_ndcg_score

if __name__ == "__main__":
    for num in range(12):
        numstring = str(num + 1)
        TensorX_approximation = reconstruct_tensor(numstring)
        groundRating_mat = get_testRating_mat()
        approximate_Rating_mat = predict_score_rank(TensorX_approximation, 104, 300, 300, alpha=0.5)
        DCG_rank_mat = get_DCG(groundRating_mat, approximate_Rating_mat, 100)
        avg_ndcg_score = get_NDCG(DCG_rank_mat)
        print(avg_ndcg_score)
    # fig1 = plt.figure()
    # fig2 = plt.figure()
    # ax = Axes3D(fig1)
    # ax2 = Axes3D(fig2)
    # t = np.arange(1000)
    # X, Y = np.meshgrid(t, t)
    # ax.plot_wireframe(X, Y, approximate_Rating_mat, rstride=1, cstride=1, cmap='rainbow')
    # ax2.plot_surface(X, Y, groundRating_mat, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
