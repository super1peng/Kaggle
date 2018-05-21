#coding:utf-8

'''
随机森林是利用多棵树进行训练并预测的一种分类器
随机森林的构建：
    1. 数据的随机性化
    2. 待选特征的随机性化

1. 数据的随机性化：
    （有放回的准确率在：70% 以上， 无放回的准确率在：60% 以上）
(1) 采用有放回的抽样方式构造子数据集，保证不同子集之间的数量级一样（不同子集／同一子集 之间的元素可以重复）
(2) 利用子数据集来构建决策树，将这个数据放到每个子决策树中，每个子决策树输出一个结果
(3) 然后根据结合策略，得到最终的分类结果这就是随机森林的输出结果

2. 待选特征的随机性化：
(1) 子树从所有的待选特征中随机选取一定的特征。
(2) 在选取的特征中选取最优的特征。

适用数据范围：数值型和标称型

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import time
import os


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
    
    '''
    n_components>=1
      n_components=NUM   设置占特征数量
    0 < n_components < 1
      n_components=0.99  设置阈值总方差占比

    whiten ：
        判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.
        对于PCA降维本身来说，一般不需要白化。
        如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化

    svd_solver：
        即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，
        一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。
            randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 
            full则是传统意义上的SVD，使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，
            arpack直接使用了scipy库的sparse SVD实现。
            默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。
    
    '''

    pca = PCA(n_components=COMPONENT_NUM, random_state=34)
    data_pca = pca.fit_transform(data)

    # 显示 pca 方差大小、方差占比、特征数量
    print(pca.explained_variance_, '\n', pca.explained_variance_ratio_, '\n', pca.n_components)
    print(sum(pca.explained_variance_ratio_))

    # 将模型进行存储
    storeModel(data_pca, 'Result_sklearn_rf.pcaData')
    return data_pca

# 构建训练模型
def trainModel(X_train, y_train):
    print("Train RF ...")
    clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=34,
    )

    '''
    n_estimators: 
        也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
        一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，
        并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值。默认是100。

    oob_score :
    即是否采用袋外样本来评估模型的好坏。默认识False。个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。

    criterion: 
        即CART树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。
        分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。
        回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。
        一般来说选择默认的标准就已经很好的。

    1) RF划分时考虑的最大特征数max_features: 
        默认是"auto",意味着划分时最多考虑N‾‾√个特征；
        如果是"log2"意味着划分时最多考虑log2N个特征；
        如果是"sqrt"或者"auto"意味着划分时最多考虑N‾‾√个特征。
        如果是整数，代表考虑的特征绝对数。
        如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。
        一般我们用默认的"auto"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

    2) 决策树最大深度max_depth: 
        默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。
        一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。
        常用的可以取值10-100之间。

    3) 内部节点再划分所需最小样本数min_samples_split: 
        这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 
        默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

    4) 叶子节点最少样本数min_samples_leaf: 
        这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 
        默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

    5）叶子节点最小的样本权重和min_weight_fraction_leaf：
        这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 
        默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

    6) 最大叶子节点数max_leaf_nodes: 
        通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
        如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，
        具体的值可以通过交叉验证得到。

    7) 节点划分最小不纯度min_impurity_split:
        这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
        一般不推荐改动默认值1e-7。
    '''




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

# 存储模型
def storeModel(model, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)

# 加载模型
def getModel(filename):
    import pickle    
    fr = open(filename, 'rb')
    return pickle.load(fr)

# 结果输出保存
def saveResult(result, csvName):
    i = 0
    fw = open(csvName, 'w')
    with open('../../Kaggle_data/digits/sample_submission.csv') as pred_file:
        fw.write('{},{}\n'.format('ImageId', 'Label'))
        for line in pred_file.readlines()[1:]:
            splits = line.strip().split(',')
            fw.write('{},{}\n'.format(splits[0], result[i]))
            i += 1
    fw.close()
    print('Result saved successfully...')

# 训练决策树模型
def trainRF():
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
    rfClf = trainModel(X_train, y_train)
    # 保存结果
    storeModel(data_pca[len(train_data):], 'Result_sklearn_rf.pcaPreData')
    storeModel(rfClf, 'Result_sklearn_rf.model')

    # 模型准确率
    y_predict = rfClf.predict(X_test)
    printAccuracy(y_test, y_predict)

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))

# 预测决策树模型
def preRF():
    startTime = time.time()
    # 加载模型和数据
    clf=getModel('Result_sklearn_rf.model')
    pcaPreData = getModel('Result_sklearn_rf.pcaPreData')

    # 结果预测
    result = clf.predict(pcaPreData)

    # 结果的输出
    saveResult(result, 'Result_sklearn_rf.csv')
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))


def main():
    # 训练并保存模型
    trainRF()

    # 通过模型预测结果
    preRF()

if __name__ == "__main__":
    main()