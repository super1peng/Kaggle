#coding:utf-8

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets


# 载入数据
digits = datasets.load_digits()
print(digits.data.shape) # 特征空间维度
print(digits.target.shape) # 标签的维度

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=30)

# 建立模型
estimator = LGBMClassifier(
    objective='multiclass',  # 多分类问题   二分类问题则是 binary
    num_leaves=31,     # 设置叶子个数
    learning_rate= 0.1,   # 设置学习率
    n_estimators= 10,  # 设置决策树的格式 默认是10个
    subsample_for_bin=1,
    subsample=1,  # 与xgboost相同

    metric = 'logloss',            # 评估指标
    silent =True,                  # 输出中间过程
    reg_alpha=0.0,                 # L1正则化系数
    min_split_gain=0.0,            # 默认0，分裂最小权重
    # early_stopping_rounds=50       # 提前终止训练
)

# 进行网格搜索
param_grid = {
    'num_leaves': list(range(10, 35, 5)),   # 找到最合适的参数
}

gs = GridSearchCV(estimator,              # 分类器
                  param_grid,             # 参数字典
                  scoring='neg_log_loss', # 评价标准
                  cv=3,                   # 三折交叉验证
                  verbose = 2,            # 打印全部中间过程（0/1/2）
                  n_jobs=1,                 # 并行计算CPU个数
                  refit=True,
)

gs.fit(x_train,y_train)
print('最佳参数:',gs.best_params_)
print('最优分数:',gs.best_score_)

# 训练模型
# lgbm = gs.best_estimator_
# lgbm.fit(x_train,y_train)
# print(lgbm.best_iteration_)
# 模型属性
# print('best_score:', lgbm.best_score_)    # 最优分数
# print('best_iteration', lgbm.best_iteration_)   # 最佳迭代器个数

# 下面进行模型的预测
# y_pred = lgbm.predict(x_test, num_iteration=lgbm.best_iteration_)

# 性能评估
# print('模型的准确率为:', accuracy_score(y_test,y_pred))