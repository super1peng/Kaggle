#coding:utf-8

# 导入必要的数据库
from sklearn import datasets
from sklearn.model_selection import train_test_split

# xgboost库
from xgboost import XGBClassifier

# 性能度量
from sklearn.metrics import accuracy_score

# 导入数据 digits数据集
digits = datasets.load_digits()
print(digits.data.shape) # 特征空间维度
print(digits.target.shape) # 标签的维度

# 将数据进行分割
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=30)


# 建立模型
model = XGBClassifier()  # 载入模型

'''
XGBClassifier
    learning_rate 设置学习率
    n_estimators  设置树的规模
    max_depth 设置树的深度
    min_child_weight 叶子节点最小权重
    gamma  惩罚项中叶子结点个数前的参数
    subsample 随机选择80%的样本建立决策树
    colsample_btree=0.8  随机选择80%的特征建立决策树
    objective='multi:softmax  损失函数的设置
    scale_pos_weight=1  解决样本个数不平衡问题
    random_state 随机数
'''


model.fit(x_train, y_train)   # 训练模型
'''
    fit 参数设置
        eval_set 评估数据集 list类型
        eval_metric 评估标准  （在多分类问题中，使用mlogloss作为损失函数）
        early_stopping_rounds = 10  如果模型的loss在10次内没有减小，那么提前结束模型的训练
        
'''


y_pred = model.predict(x_test)  # 模型预测

# 性能度量
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

# 混淆矩阵
from sklearn import metrics

confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)

# 输出特征重要性
import matplotlib.pyplot as plt
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,15))
plot_importance(model, height=0.5, max_num_features=64, ax=ax)

plt.show()

    
        