import numpy as np

pathdir = "E:/PYworkspace/EETDR"


def get_infile(infile=pathdir + "/data/yelp_recursive_data/test.entry"):
    return infile


def load_data(num_users, num_items):
    """
    读取文件，获得用户，产品，特征的总数，并且得到规范化的数据
    :param num_users:测试集需要的最大用户id
    :param num_items：测试集需要的最大产品id
    :return:
    """
    print(get_infile())
    ground_true_test_rating_mat = np.zeros((num_users, num_items))
    with open(get_infile(), "r") as infile:
        print("加载数据......")
        for line in infile.readlines():
            line = line.split(",")
            if int(line[0]) < num_users and int(line[1]) < num_items:
                ground_true_test_rating_mat[int(line[0])][int(line[1])] = int(line[2])
    return ground_true_test_rating_mat


def build_1000test_data_mat():
    x = load_data(1000, 1000)
    np.save(pathdir+"/data/testdata/1000test_mat",x)
    return x


if __name__ == "__main__":
    print(build_1000test_data_mat())
