#coding:utf-8
'''
构造一个能识别数据 0-9 的基于knn分类器的手写数组识别系统
数据是 宽高为 32像素*32像素的黑白图像
'''

import csv
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
def opencsv():
    train = pd.read_csv('../../Kaggle_data/digits/train.csv')
    test = pd.read_csv('../../Kaggle_data/digits/test.csv')

    train_data = train.values[0:, 1:]
    train_label = train.values[0:, 0]

    test_data = test.values[0:,0:]
    return train_data,train_label,test_data

def saveResult(result, csvName):
    with open(csvName, 'w',newline='') as myFile: # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
    #python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
        myWriter = csv.writer(myFile) # 对文件执行写入
        myWriter.writerow(["ImageId", "Label"]) # 设置表格的列名
        index = 0
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            # tmp.append(i)
            tmp.append(int(i)) # 测试集的标签值
            myWriter.writerow(tmp)

def knnClassify(trainData,trainLabel):
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData, np.ravel(trainLabel)) # 将多维数组降成一维
    return knnClf

def Recognition_knn():
    start_time = time.time()

    # 加载数据
    trainData, trainLabel, testData = opencsv()
    print("trainData ==>", type(trainData), np.shape(trainData))
    print("trainLable ==>", type(trainLabel), np.shape(trainLabel))
    print("testData ==>", type(testData), np.shape(testData))

    print("Data Load Finish")
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time)) 


    # 模型训练
    knnClf = knnClassify(trainData, trainLabel)

    # knn 模型不需要进行训练
    # 预测结果
    testLabel = knnClf.predict(testData)

    # 将结果输出
    saveResult(testLabel,'Result_sklearn_knn.csv')
    print("Finish!")
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - stop_time_l))
if __name__ == "__main__":
    Recognition_knn()