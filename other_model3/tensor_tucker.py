# Tensor Multi-task 9/11/2017 NanWang
import math
import random
import time

import numpy as np
from sklearn.cluster import KMeans
import tensor_sparse_multi_tasks_all_diff as tsmtr

pathdir = "E:/PYworkspace/EETDR/"


def get_index(key):
    index = key[1:-1].split(',')
    for i in range(3):
        index[i] = int(index[i])
    return index


fin_uif_train_entry = open('E:/PYworkspace/MTER/yelp_restaurant_recursive_entry_sigir/yelp_recursive_train.entry',
                           encoding='UTF-8')
fin_uif_test_entry = open(pathdir + 'data/yelp_restaurant_recursive_entry_sigir/yelp_recursive_test.entry',
                          encoding='UTF-8')
fin_uifw_train_entry = open(pathdir + 'data/yelp_restaurant_recursive_entry_sigir/yelp_recursive_train.uifwords_entry',
                            encoding='UTF-8')
fin_uifw_test_entry = open(pathdir + 'data/yelp_restaurant_recursive_entry_sigir/yelp_recursive_test.uifwords_entry',
                           encoding='UTF-8')
fin_feature_map = open(pathdir + 'data/yelp_restaurant_recursive_entry_sigir/yelp_recursive.featuremap',
                       encoding='UTF-8')
fin_word_map = open(pathdir + 'data/yelp_restaurant_recursive_entry_sigir/word.map_sentiment', encoding='UTF-8')

fout_itemrank = open(pathdir + '/result/yelp_itemrec_NEW_bpr_2000.reclist', 'w')
fout_rec_explanations = open(pathdir + '/result/yelp_itemrec_NEW_bpr_2000.explanation', 'w')

U_dim = 24
I_dim = 12
F_dim = 12
W_dim = 24

num_iter = 2000  # should be around 200k

feature_maplines = fin_feature_map.readlines()
feature_map = {}

# create feature and words maping files
for line in feature_maplines:
    eachline = line.strip().split('=')
    feature_map[eachline[0]] = eachline[1]
word_senti = {}
word_maplines = fin_word_map.readlines()
word_map = {}
for line in word_maplines:
    eachline = line.strip().split('=')
    word_map[eachline[0]] = eachline[1]
    word_senti[int(eachline[0])] = int(eachline[2])
U_num = 10719
I_num = 10410
F_num = 104
W_num = 1019

# if_list = []
# for line in fin_uifw_train_entry.readlines():
#     se = line.strip().split(",")
#     if [se[1], se[2]] not in if_list:
#         if_list.append([se[1], se[2]])
#         W_IF[str([se[1], se[2]]),]
evulate_res = []
sps_tensor_useritemf = {}
sps_tensor_useritemf_test = {}
sps_tensor_userwordf = {}
sps_tensor_ifw = {}
sps_tensor_useritemfw_test = {}
useritemfeature_test = {}
if_pair_list = []  # 一对一对
if_pair_list2 = list(range(I_num))  # 列表数组
rec_expl_output = []
rec_word_output = []

# overall_rating_train = np.zeros((U_num, I_num))  # 只是为了写文件保存，npy格式
# overall_rating_test = np.zeros((U_num, I_num))  # 只是为了写文件保存，npy格式
sps_overall_rating_train = {}
sps_overall_rating_test = {}
rec_item = np.zeros((U_num, I_num))
item_feature_mentioned = np.zeros((I_num, F_num))
item_feature_mentioned_test = np.zeros((I_num, F_num))
feature_word_used = np.zeros((F_num, W_num))
item_feature_num_record = np.zeros((I_num, 1))

cnt_train = 0
cnt_test = 0

# read user-item-feature entries （1）构建UIF稀疏张量
uif_train_lines = fin_uif_train_entry.readlines()
uif_test_lines = fin_uif_test_entry.readlines()
for line in uif_train_lines:
    eachline = line.strip().split(',')
    ft_sent_pair = eachline[3].strip()
    if ft_sent_pair != '':
        u_idx = int(eachline[0])
        i_idx = int(eachline[1])
        over_rating = int(eachline[2])
        if over_rating == 0:
            over_rating = 1
        f_s_pairs = ft_sent_pair.strip().split(' ')
        feature_list = []
        for f_s in f_s_pairs:
            fea = f_s.strip().split(':')
            f_idx = int(fea[0])
            # user_feature_attention[u_idx][f_idx] += 1
            senti = int(fea[1])
            if senti not in feature_list:
                feature_list.append(f_idx)
            # item_feature_quality[i_idx][f_idx] += senti
            if str([u_idx, i_idx, f_idx]) not in sps_tensor_useritemf:
                sps_tensor_useritemf[str([u_idx, i_idx, f_idx])] = 0
                cnt_train += 1
            sps_tensor_useritemf[str([u_idx, i_idx, f_idx])] += senti
            if item_feature_mentioned[i_idx][f_idx] == 0:
                item_feature_num_record[i_idx] += 1
            item_feature_mentioned[i_idx][f_idx] += 1
        if_pair_list2[i_idx] = feature_list
        # overall_rating_train[u_idx][i_idx] = over_rating
        sps_overall_rating_train[str([u_idx, i_idx])] = over_rating
        # 不加overrating项
        # sps_tensor_useritemf[str([u_idx, i_idx, F_num])] = over_rating

# np.save("E:\PYworkspace\EETDR\data\ex_stroe_data\\overall_rating_train", overall_rating_train)

for line in uif_test_lines:
    eachline = line.strip().split(',')
    ft_sent_pair = eachline[3].strip()
    if ft_sent_pair != '':
        u_idx = int(eachline[0])
        i_idx = int(eachline[1])
        over_rating = int(eachline[2])
        if over_rating == 0:
            over_rating = 1
        f_s_pairs = ft_sent_pair.strip().split(' ')
        for f_s in f_s_pairs:
            fea = f_s.strip().split(':')
            f_idx = int(fea[0])
            senti = int(fea[1])

            if str([u_idx, i_idx, f_idx]) not in sps_tensor_useritemf_test:
                sps_tensor_useritemf_test[str([u_idx, i_idx, f_idx])] = 0
                cnt_test += 1
            sps_tensor_useritemf_test[str([u_idx, i_idx, f_idx])] += senti
            if item_feature_mentioned_test[i_idx][f_idx] == 0:
                item_feature_num_record[i_idx] += 1
            item_feature_mentioned_test[i_idx][f_idx] += 1
        # overall_rating_test[u_idx][i_idx] = over_rating
        sps_overall_rating_test[str([u_idx, i_idx])] = over_rating
        sps_tensor_useritemf_test[str([u_idx, i_idx, F_num])] = over_rating
# np.save("E:\PYworkspace\EETDR\data\ex_stroe_data\\overall_rating_test", overall_rating_test)

# 除了第104个特征，也就是over_rating外全都sigmoid处理(暂时没加over_rating项)
for key in sps_tensor_useritemf.keys():
    index = get_index(key)
    if index[2] != F_num:
        sps_tensor_useritemf[key] = 1 + 4 / (1 + np.exp(0 - sps_tensor_useritemf[key]))
for key in sps_tensor_useritemf_test.keys():
    index = get_index(key)
    if index[2] != F_num:
        sps_tensor_useritemf_test[key] = 1 + 4 / (1 + np.exp(0 - sps_tensor_useritemf_test[key]))

# read user/item-feture-word entries（2）构建UFO和UFW稀疏张量
uifw_train_lines = fin_uifw_train_entry.readlines()
uifw_test_lines = fin_uifw_test_entry.readlines()
for line in uifw_train_lines:
    eachline = line.strip().split(',')
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    f_idx = int(eachline[2])
    w_idx = int(eachline[3])
    w_senti = word_senti[w_idx]

    feature_word_used[f_idx][w_idx] += 1

    if str([u_idx, f_idx, w_idx]) not in sps_tensor_userwordf:
        sps_tensor_userwordf[str([u_idx, f_idx, w_idx])] = 0
    if sps_tensor_userwordf[str([u_idx, f_idx, w_idx])]==0:
        sps_tensor_userwordf[str([u_idx, f_idx, w_idx])] += 1

    if str([i_idx, f_idx, w_idx]) not in sps_tensor_ifw:
        sps_tensor_ifw[str([i_idx, f_idx, w_idx])] = 0
    sps_tensor_ifw[str([i_idx, f_idx, w_idx])] += 1
    if_pair_list.append((i_idx, f_idx))

# 去重
if_pair_list = list(set(if_pair_list))
# sigmoid word
for key in sps_tensor_ifw.keys():
    index = get_index(key)
    sps_tensor_ifw[key] = 1 + 4 * (2 / (1 + np.exp(0 - sps_tensor_ifw[key])) - 1)
    # sps_tensor_ifw[key] = 1 + 4 / (1 + np.exp(0 - sps_tensor_ifw[key]))
    sps_tensor_ifw[key] *= word_senti[index[2]]

for line in uifw_test_lines:
    cnt_train += 1
    eachline = line.strip().split(',')
    u_idx = int(eachline[0])
    i_idx = int(eachline[1])
    f_idx = int(eachline[2])
    w_idx = int(eachline[3])
    # feature_word_used[f_idx][w_idx] += 1

    if str([u_idx, i_idx, f_idx, w_idx]) not in sps_tensor_useritemfw_test:
        sps_tensor_useritemfw_test[str([u_idx, i_idx, f_idx, w_idx])] = 0
    sps_tensor_useritemfw_test[str([u_idx, i_idx, f_idx, w_idx])] += 1
    useritemfeature_test[(u_idx, i_idx, f_idx)] = 1

print("train size:" + str(cnt_train) + '\n')
print("test size:" + str(cnt_test) + '\n')

# training
start_time = time.time()
(U, I, F, W,) = tsmtr.learn_HAT_SGD_adagrad(sps_tensor_useritemf,
                                            sps_tensor_ifw,
                                            sps_overall_rating_train,
                                            U_dim, I_dim, F_dim,
                                            W_dim, U_num, I_num,
                                            F_num + 1, W_num,
                                            num_iter, lr=0.1,
                                            dis=False,
                                            cost_function='abs',
                                            random_seed=0, eps=1e-8)
train_time = time.time() - start_time
# generating item recommendation lists
print('Generating item recommendation ranking lists and explanations ...')

rec_item = np.einsum('ma,na ->mn ', U, np.hstack((I, np.tile(F[104], (I_num, 1)))))
# item_feature_word_vec = tsmtr.construct_IF_W_row(I, F, W, [0, 1])
# u_2_i_f_preference = tsmtr.construct_IF_U_row(if_pair_list, I, F, U, 5, 5)

for item in sps_overall_rating_test.items():
    index = item[0][1:-1].split(',')
    user_id = int(index[0])
    item_id = int(index[1])
    real_rating = item[1]
    rec_rating = rec_item[user_id][item_id]
    evulate_res.append([user_id, item_id, real_rating, rec_rating])
from EETDR.metrics import metric

metric = metric.Metric()
print("MAE:")
print(metric.MAE(evulate_res))
print("RMSE:")
print(metric.RMSE(evulate_res))
print(train_time)

list_length = 100
top_feature_num = 5
rec_item_num = 0
temp_feature_vector = []

for i in range(U_num):
    print(i)
    rec_item_num = 0
    itemrec_for_user = np.zeros((I_num, 1))
    purchased = 0
    # 训练集中的数据不在推荐列表中展示
    for jj in range(I_num):
        if str([i, jj]) in sps_overall_rating_train.keys():
            itemrec_for_user[jj] = 0
        else:
            itemrec_for_user[jj] = rec_item[i, jj]
    temp = {}
    temp_list = []
    top_item = 0
    flag = 1
    while flag > 0:
        top_item = np.where(itemrec_for_user == np.max(itemrec_for_user))[0][0]
        # 最大值是0了，停止推荐
        if itemrec_for_user[top_item] == 0:
            flag = 0
        # 如果小于100个继续推荐
        elif rec_item_num < list_length:
            rec_item_num += 1
            # temp 存储推荐的物品的评分
            if str([i, top_item]) in sps_overall_rating_test.keys():
                temp[top_item] = sps_overall_rating_test[str([i, top_item])]
                purchased = 1
            else:
                temp[top_item] = 0
            # temp_list 推荐列表
            temp_list.append(top_item)
            itemrec_for_user[top_item] = 0
        else:
            flag = 0
    top_item_num = 0
    if purchased == 1:
        rec_expl_output.append("#User:" + str(i) + ", Recommended item list:" + '\n')
        fout_itemrank.write('###User:' + str(i) + ' reclist: \n')
        for key in temp_list:
            temp_feature_vector = []
            top_item_num += 1
            if str([i, key]) in sps_overall_rating_test.keys():
                fout_itemrank.write(
                    '@Purchased item: %-8d' % (key) + ' Real Overall_rating: %-8d' % (temp[key]) + '\n')
                # evulate_res.append([i, key, temp[key], rec_item[i][key]])
                if top_item_num <= 20:
                    rec_expl_output.append("@Purchased item:" + str(key) + " User_preference:" + str(
                        rec_item[i][key]) + '\n' + "Rec_features: " + "\n")
                    u_2_i_f_preference = tsmtr.construct_IF_U_row(U, I, F, i, key, if_pair_list2[key])

                    # 不设置top feature number:
                    if_top_word_list = []
                    for feature in u_2_i_f_preference:
                        rec_expl_output.append("\t" + feature_map[str(feature)])
                        rec_expl_output.append(":" + str(u_2_i_f_preference[feature]) + "  @word:")
                        item_feature_word_vec = tsmtr.construct_IF_W_vec(I, F, W, [key, feature])
                        for tt in range(13):
                            top_word_score = np.max(item_feature_word_vec)
                            top_word = np.where(item_feature_word_vec == top_word_score)[0][0]
                            item_feature_word_vec[top_word] = 0
                            # if_top_word_list.append((top_word, top_word_score))
                            if tt > 3:
                                # rec_expl_output.append(" " + word_map[str(top_word)] + "/" + str(top_word_score))
                                rec_expl_output.append(" " + word_map[str(top_word)] )
                        rec_expl_output.append("\n")

                    rec_expl_output.append('\n')

            else:
                fout_itemrank.write('@Rec item: %-8d' % (key) + ' @Rec rating: %-8d' % (temp[key]) + '\n')
                if top_item_num <= 20:
                    rec_expl_output.append("@Recommended item:" + str(key) + " User_preference:" + str(
                        rec_item[i][key]) + '\n' + "Rec_features: " + "\n")
                    u_2_i_f_preference = tsmtr.construct_IF_U_row(U, I, F, i, key, if_pair_list2[key])
                    # 不设置top feature number:
                    if_top_word_list = []
                    for feature in u_2_i_f_preference:
                        rec_expl_output.append("\t" + feature_map[str(feature)])
                        rec_expl_output.append(":" + str(u_2_i_f_preference[feature]) + "  @word ")
                        item_feature_word_vec = tsmtr.construct_IF_W_vec(I, F, W, [key, feature])
                        for tt in range(13):
                            top_word_score = np.max(item_feature_word_vec)
                            top_word = np.where(item_feature_word_vec == top_word_score)[0][0]
                            item_feature_word_vec[top_word] = 0
                            # if_top_word_list.append((top_word, top_word_score))
                            if tt>3:
                                rec_expl_output.append(" " + word_map[str(top_word)])
                            # rec_expl_output.append(" " + word_map[str(top_word)] + "/" + str(top_word_score))
                        rec_expl_output.append("\n")

fout_rec_explanations.writelines(rec_expl_output)
