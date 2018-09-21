def get_entyfile(entyfile="E:/PYworkspace/EETDR/data/yelp_recursive_data/all.entry"):
    return entyfile


def load_data():
    """
    读取文件，获得用户，产品，特征的总数，并且得到规范化的数据
    :return:
    """
    trainlist = []
    testlist = []
    trainfile = open("E:/PYworkspace/EETDR/data/yelp_recursive_data/New_train.txt","a")
    testfile = open("E:/PYworkspace/EETDR/data/yelp_recursive_data/New_test.txt","a")
    with open(get_entyfile(), "r") as infile:
        print("加载数据......")
        for line in infile.readlines():
            line2 = line.split(",")
            if line2[0] not in trainlist:
                trainlist.append(line2[0])
                trainfile.write(line)
            else:
                if line2[0] not in testlist:
                    testlist.append(line2[0])
                    testfile.write(line)
                else:
                    trainfile.write(line)
        print("拆分完成")

load_data()