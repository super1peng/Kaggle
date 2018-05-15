#coding:utf-8

'''
使用SVM进行多分类任务，有以下三种方式：
    1：一对多法 (one versus rest OVR)
        训练时，依次把某个类别的样本归为一类，其他剩余的样本归为一类，这样k个类别的样本就构造了k个svm。
        预测时，将未知样本分类为具有最大分类函数值的那类。
    2：一对一法（one-versus-one)
        其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。
        当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。
    3：层次支持向量机
        层次分类法首先将所有类别分成两个子类，再将子类进一步划分成两个次级子类，如此循环，直到得到一个单独的类别为止。
'''
import time
import csv
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def opencsv():
    dataTrain = pd.read_csv('../../Kaggle_data/digits/train.csv')
    dataPre = pd.read_csv('../../Kaggle_data/digits/test.csv')

    trainData = dataTrain.values[:, 1:]  # 读入全部训练数据
    trainLabel = dataTrain.values[:, 0]
    
    preData = dataPre.values[:, :]  # 测试全部测试个数据
    return trainData, trainLabel, preData

# 数据预处理 降维  主成分分析
def dRCsv(x_train, x_test, preData, COMPONENT_NUM):
    print('dimensionality reduction...')

    trainData = np.array(x_train)
    testData = np.array(x_test)
    preData = np.array(preData)

    # 进行主成分分析
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(trainData)
    pcaTrainData = pca.transform(trainData)  # trainData 在 X 上完成降维
    pcaTestData = pca.transform(testData)  # testData 在 X 上完成降维
    pcaPreData = pca.transform(preData)   # preData 在 X 上完成降维
    
    # pca 方差大小、方差占比、特征数量
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components_)
    print(sum(pca.explained_variance_ratio_))
    return pcaTrainData,  pcaTestData, pcaPreData

# 训练模型
def trainModel(trainData, trainLabel):
    print('Train SVM...')
    svmClf = SVC(C=4, kernel='rbf') # 设置惩罚参数 和 核函数--高斯径向基核函数
    svmClf.fit(trainData, trainLabel)  # 训练SVM
    return svmClf

# 结果输出保存
def saveResult(result, csvName):
    with open(csvName, 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index, int(r)])
    print('Saved successfully...')  # 保存预测结果

# 对数据进行分析
def analyse_data(dataMat):
    meanVals = np.mean(dataMat, axis=0) # 求出每列的平均值
    meanRemoved = dataMat - meanVals # 中心化 每一列的特征值 - 每一列的平均值
    covMat = np.cov(meanRemoved, rowvar=0) # 计算协方差值

    eigvals, eigVects = np.linalg.eig(np.mat(covMat)) # 进行特征值分解 得到 特征值 和 特征向量
    eigValInd = np.argsort(eigvals) #  argsort 对特征值进行排序，返回的是数值从小到大的索引值

    topNfeat = 100 # 需要保留的特征维度，即要压缩成的维度数

    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 选取前 topN大的特征值
    # 计算所有特征值的总和
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0

    for i in range(0, len(eigValInd)): 
        # 被选择的特征值进行相加
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        '''
        我们发现其中有超过20%的特征值都是0。
        这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。
        最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。
        这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。
        最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.
        '''
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))
    print("主成分分析结束")
    return 0


def getOptimalAccuracy(trainData, trainLabel, preData):
    x_train, x_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.1) # 将训练数据分为 训练集和验证集
    lineLen, featureLen = np.shape(x_test) 
    
    minErr = 1
    minSumErr = 0
    optimalNum = 1
    optimalLabel = []
    optimalSVMClf = None
    pcaPreDataResult = None
    for i in range(30, 45, 1):   # 将数据降到不同的特征维度
        # 评估训练结果
        pcaTrainData,  pcaTestData, pcaPreData = dRCsv(x_train, x_test, preData, i)
        svmClf = trainModel(pcaTrainData, y_train)
        svmtestLabel = svmClf.predict(pcaTestData)

        errArr = np.mat(np.ones((lineLen, 1)))
        sumErrArr = errArr[svmtestLabel != y_test].sum()
        sumErr = sumErrArr/lineLen

        print('i=%s' % i, lineLen, sumErrArr, sumErr)
        if sumErr <= minErr:
            minErr = sumErr
            minSumErr = sumErrArr
            optimalNum = i
            optimalSVMClf = svmClf
            optimalLabel = svmtestLabel
            pcaPreDataResult = pcaPreData
            print("i=%s >>>>> \t" % i, lineLen, int(minSumErr), 1-minErr)

    '''
    展现 准确率与召回率
        precision 准确率
        recall 召回率
        f1-score  准确率和召回率的一个综合得分
        support 参与比较的数量
    '''

    # target_names 以 y的label分类为准
    # target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    target_names = [str(i) for i in list(set(y_test))]
    print(target_names)
    print(classification_report(y_test, optimalLabel, target_names=target_names))
    print("特征数量= %s, 存在最优解：>>> \t" % optimalNum, lineLen, int(minSumErr), 1-minErr)
    return optimalSVMClf, pcaPreDataResult


# 存储模型
def storeModel(model, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)
    return 0


# 加载模型
def getModel(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def trainDRSVM():
    startTime = time.time()

    # 加载数据
    trainData, trainLabel, preData = opencsv()

    # 模型训练（里面加入了降维操作）
    optimalSVMClf, pcaPreData = getOptimalAccuracy(trainData,trainLabel, preData)
    
    storeModel(optimalSVMClf, 'Result_sklearn_SVM.model')
    storeModel(pcaPreData, 'Result_sklearn_SVM.pcaPreData')

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))

    return 0 

def preDRSVM():
    startTime = time.time()
    # 加载模型和数据
    optimalSVMClf = getModel('Result_sklearn_SVM.model')
    pcaPreData = getModel('Result_sklearn_SVM.pcaPreData')

    # 结果预测
    testLabel = optimalSVMClf.predict(pcaPreData)
    # print("testLabel = %f" % testscore)
    # 结果的输出
    saveResult(testLabel, 'Result_sklearn_SVM.csv')
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))
    return 0


if __name__ == "__main__":
    trainData, trainLabel, preData = opencsv()
    trainDRSVM()
    # analyse_data(trainData)
    # preDRSVM()