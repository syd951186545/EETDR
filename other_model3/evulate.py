from sklearn.model_selection import KFold  # K折交叉验证
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差

# calculate AUC
import pylab as pl
from math import log,exp,sqrt
import itertools
import operator

def read_file(file_path, accuracy=2):
    db = []  #(score,nonclk,clk)
    pos, neg = 0, 0 #正负样本数量
    #读取数据
    with open(file_path,'r') as fs:
        for line in fs:
            temp = eval(line)
            #精度可控
            #score = '%.1f' % float(temp[0])
            score = float(temp[0])
            trueLabel = int(temp[1])
            sample = [score, 0, 1] if trueLabel == 1 else [score, 1, 0]
            score,nonclk,clk = sample
            pos += clk #正样本
            neg += nonclk #负样本
            db.append(sample)
    return db, pos, neg

def get_roc(db, pos, neg):
    #按照输出概率，从大到小排序
    db = sorted(db, key=lambda x:x[0], reverse=True)
    file=open('data.txt','w')
    file.write(str(db))
    file.close()
    #计算ROC坐标点
    xy_arr = []
    tp, fp = 0., 0.
    for i in range(len(db)):
        tp += db[i][2]
        fp += db[i][1]
        xy_arr.append([fp/neg,tp/pos])
    return xy_arr

def get_AUC(xy_arr):
    #计算曲线下面积
    auc = 0.
    prev_x = 0
    for x,y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    return auc

def draw_ROC(xy_arr):
    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    pl.title("ROC curve of %s (AUC = %.4f)" % ('clk',auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot(x, y)# use pylab to plot x and y
    pl.show()# show the plot on the screen