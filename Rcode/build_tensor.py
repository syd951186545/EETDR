import numpy as np
import tensorly as tl
import tensorflow as tf
from scipy import sparse as sp
import math

pathdir = "E:/PYworkspace/EETDR"


class Build_tensor:

    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.num_users = 0
        self.num_items = 0
        self.num_features = 0
        self.num_datas = 0
        self.pieces = []
        self.load_data()

    def load_data(self):
        """
        读取文件，获得用户，产品，特征的总数，并且得到规范化的数据
        :return:
        """
        with open(self.infile, "r") as infile:
            print("加载数据......")
            for line in infile.readlines():
                feature_list = []
                line = line.split(",")
                line[3] = line[3].replace("\n", "")
                for featureS in line[3].split(" "):
                    templist = featureS.split(":")
                    feature_list.append((int(templist[0]), int(templist[1])))
                    if self.num_features < int(templist[0]):
                        self.num_features = int(templist[0])
                if self.num_users < int(line[0]):
                    self.num_users = int(line[0])
                if self.num_items < int(line[1]):
                    self.num_items = int(line[1])
                self.pieces.append([int(line[0]), int(line[1]), int(line[2]), feature_list])
                self.num_datas += 1
            print("加载完成，共{}条数据".format(self.num_datas))

    def build_tensor(self, tensor=False, matix=False, sprasefile=False):

        print("正在构建张量......")
        TensorX = np.ndarray(shape=(self.num_users + 1, self.num_items + 1, self.num_features + 1), dtype=np.float16)
        # TensorX = tl.zeros(shape=(self.num_users,self.num_items,self.num_features))
        # TensorX = tf.SparseTensor()
        # TensorX = sp.coo_matrix(shape=(self.num_users,self.num_items,self.num_features))
        for piece in self.pieces:
            for feature in piece[3]:
                TensorX[piece[0]][piece[1]][feature[0]] += np.float(feature[1])
        print("张量构建完成")
        print("稀疏形式存储......")
        Sparselist = []
        Sparsemat = []
        i_index, j_index, k_index = TensorX.nonzero()
        if sprasefile:
            with open(self.outfile, "w") as outf:
                for count in range(len(i_index)):
                    outf.write("{},{},{},{}\n".format(i_index[count], j_index[count], k_index[count],
                                                      TensorX[i_index[count]][j_index[count]][k_index[count]]))
            print("完成存储非零值索引{}条".format(len(i_index)))

        if matix:
            for count in range(len(i_index)):
                Sparselist.append(i_index[count])
                Sparselist.append(j_index[count])
                Sparselist.append(k_index[count])
                Sparselist.append(TensorX[i_index[count]][j_index[count]][k_index[count]])

            Sparsemat = np.array(Sparselist, dtype=np.int16)
            Sparsemat = Sparsemat.reshape((len(i_index), 4))

        if sprasefile:
            print("存储csv文件{}".format(self.outfile))
        if matix:
            print("返回稀疏矩阵")
            if tensor:
                print("返回全张量")
                return TensorX, Sparsemat
            np.save("E:/PYworkspace/EETDR/result/X", Sparsemat)
            return Sparsemat
        if tensor:
            print("返回全张量")
            return TensorX
        return

    def build_UA(self):
        print("构建user-aspect真实矩阵......")
        UA = np.zeros(shape=(self.num_users + 1, self.num_features + 1))
        for piece in self.pieces:
            for feature in piece[3]:
                UA[piece[0]][feature[0]] += 1
        print("构建完成")
        return UA

    def build_IA(self):
        print("构建item-aspect真实矩阵......")
        IA = np.zeros(shape=(self.num_users + 1, self.num_features + 1))
        for piece in self.pieces:
            for feature in piece[3]:
                IA[piece[1]][feature[0]] += 1
        print("构建完成")
        return IA

    def build_part_tensor(self):
        print("正在构建张量......")
        countt = 0
        TensorX = np.ndarray(shape=(1000, 1000, 105), dtype=np.float)
        # TensorX = tl.zeros(shape=(self.num_users,self.num_items,self.num_features))
        # TensorX = tf.SparseTensor()
        # TensorX = sp.coo_matrix(shape=(self.num_users,self.num_items,self.num_features))
        for piece in self.pieces:
            for feature in piece[3]:
                if piece[0] < 1000 and piece[1] < 1000:
                    TensorX[piece[0]][piece[1]][feature[0]] += np.int16(feature[1])
                    countt += 1

        print("提取{}条数据构建部分张量完成".format(countt))

        Sparselist = []
        Sparsemat = []
        i_index, j_index, k_index = TensorX.nonzero()
        for count in range(len(i_index)):
            TensorX[i_index[count]][j_index[count]][k_index[count]] = 1 + (
                    4 / (1 + math.exp(-TensorX[i_index[count]][j_index[count]][k_index[count]])))
            Sparselist.append(i_index[count])
            Sparselist.append(j_index[count])
            Sparselist.append(k_index[count])
            Sparselist.append(TensorX[i_index[count]][j_index[count]][k_index[count]])

        Sparsemat = np.array(Sparselist, dtype=np.int16)
        Sparsemat = Sparsemat.reshape((len(i_index), 4))
        print("构建user-item真实评分矩阵......")
        R = np.zeros(shape=(1000, 1000))
        for piece in self.pieces:
            if piece[0] < 1000 and piece[1] < 1000:
                R[piece[0]][piece[1]] = piece[2]
                TensorX[piece[0]][piece[1]][104] = piece[2]

        print("构建user-aspect真实矩阵......")
        UA = np.zeros(shape=(1000, self.num_features + 1))
        for piece in self.pieces:
            for feature in piece[3]:
                if piece[0] < 1000:
                    UA[piece[0]][feature[0]] += 1
        for i in range(1000):
            for k1 in range(self.num_features):
                if UA[i][k1] is not 0:
                    UA[i][k1] = 1 + (4 / (math.exp(-UA[i][k1]) + 1))
        print("构建完成")

        print("构建item-aspect真实矩阵......")
        IA = np.zeros(shape=(1000, self.num_features + 1))
        for piece in self.pieces:
            for feature in piece[3]:
                if piece[1] < 1000:
                    IA[piece[1]][feature[0]] += 1
        for j in range(1000):
            for k2 in range(self.num_features):
                if IA[j][k2] is not 0:
                    IA[j][k2] = 1 + (4 / (math.exp(-IA[j][k2]) + 1))
        print("构建完成")
        np.save("E:/PYworkspace/EETDR/data/preprodata/R1000", R)
        np.save("E:/PYworkspace/EETDR/data/preprodata/X1000", Sparsemat)
        np.save("E:/PYworkspace/EETDR/data/preprodata/UA1000", UA)
        np.save("E:/PYworkspace/EETDR/data/preprodata/IA1000", IA)
        np.save("E:/PYworkspace/EETDR/data/preprodata/TensorX1000", TensorX)
        return TensorX, UA, IA

    # def build_test_data(self):


if __name__ == "__main__":
    bt = Build_tensor(infile="E:/PYworkspace/EETDR/data/yelp_recursive_data/train.entry",
                      outfile="E:/PYworkspace/EETDR/result/UIA.sparsemat")
    TensorX, _, _ = bt.build_part_tensor()
    print(TensorX[:,:,104])
