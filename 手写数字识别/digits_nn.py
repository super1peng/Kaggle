#coding:utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import time


# 读取数据函数
def opencsv():
    train_data = pd.read_csv('../../Kaggle_data/digits/train.csv')
    test_data = pd.read_csv('../../Kaggle_data/digits/test.csv')
    
    # 将两个数据连接到一起进行处理
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    data.drop(['label'], axis=1, inplace=True)
    label = train_data.label
    return train_data, test_data, data, label


# 进行数据预处理
def dRPCA(data, COMPONENT_NUM=100):
    print('dimensionality reduction...')
    data = np.array(data)
    pca = PCA(n_components=COMPONENT_NUM, random_state=34)
    data_pca = pca.fit_transform(data)

    # 显示 pca 方差大小、方差占比、特征数量
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components)
    print(sum(pca.explained_variance_ratio_))
    return data_pca

def trainModel(X_train, y_train):
    print("Train NN ...")
    clf = MLPClassifier(
        hidden_layer_sizes=(100, ),
        activation='relu',
        alpha=0.0001,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=34,
    )
    clf.fit(X_train, y_train)
    return clf


# 计算准确率
def printAccuracy(y_test ,y_predict):
    zeroLable = y_test - y_predict
    rightCount = 0
    for i in range(len(zeroLable)):
        if list(zeroLable)[i] == 0:
            rightCount += 1
    print('the right rate is:', float(rightCount) / len(zeroLable))


def train_nn():
    start_time = time.time()

    # 加载数据
    train_data, test_data, data, label = opencsv()
    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f s' % (stop_time_l - start_time))

    startTime = time.time()
    # 模型训练 (数据预处理-降维)
    data_pca = dRPCA(data,100)


    # 对 “train_data” 划分 训练集 测试集  
    X_train, X_test, y_train, y_test = train_test_split(
        data_pca[0:len(train_data)], label, test_size=0.1, random_state=34)
    
    # 进行训练
    nnClf = trainModel(X_train, y_train)

    # 模型准确率
    y_predict = nnClf.predict(X_test)
    printAccuracy(y_test, y_predict)

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))


def main():
    train_nn()


if __name__ == "__main__":
    main()