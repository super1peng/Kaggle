#coding:utf-8
'''
KNN 工作原理：
1、假设有一个带标签的样本数据集（训练样本集），其中包含每条数据与所属类别的对应关系。
2、输入没有标签的新数据之后，将新数据的每个特征与样本集中数据对应的特征进行比较。
    2.1 计算新数据与样本数据集中每条数据的距离
    2.2 对求得的所有距离进行排序（从大到小排序，越小表示越相似）
    2.3 取前k个样本数据对应的分类标签
3、求k个数据中出现次数最多的分类标签作新数据的分类
'''

import numpy as np
import matplotlib.pyplot as plt
import operator

def classify0(inX,dataSet,labels,k):
    '''
    inX：需要进行预测的 数据
    dataSet：特征
    labels：标签
    k：取前多少个标签进行判断
    '''
    dataSetSize = dataSet.shape[0]
    # 计算误差
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    
    # 误差的平方
    sqDiffMat = diffMat**2

    # 计算误差平方和
    sqDistances = sqDiffMat.sum(axis=1)

    # 取根号
    distances = sqDistances**0.5

    # 对误差进行排序操作，默认是降序排序
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel =labels[sortedDistances[i]]
        
        # 统计每个标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    
    # 按照标签次数出现的多少进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2array(filename):
    """
    Desc:
        导入训练数据
    parameters:
        filename: 数据文件路径
    return: 
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0 
    returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


def figure(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    plt.show()
    return 0 

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)  #存放每一列的最小值，min(0)参数0可以从列中选取最小值，而不是当前行最小值
    maxVals = dataSet.max(0)  #存放每一列的最大值
    ranges = maxVals - minVals #1 * 3 矩阵
    normDataSet = np.zeros(np.shape(dataSet))   #列
    m = dataSet.shape[0]      #行
    normDataSet = dataSet - np.tile(minVals, (m, 1))  #tile(A, (row, col))  重复数组操作
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


#分类器针对约会网站的测试代码
def dataingClassTest(datingDataMat,datingLabels):
    hoRatio = 0.1
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #用于测试的数据条数
    errorCount = 0.0   #错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                                    datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d"\
              %(classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))

if __name__ == "__main__":
    filename = "./datingTestSet2.txt"
    data, label = file2array(filename)
    figure(data, label)
    dataingClassTest(data,label)