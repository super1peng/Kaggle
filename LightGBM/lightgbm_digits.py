#coding:utf-8

'''
本文件是对light gbm 的简单使用，数据集为 digits
'''

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 获取数据
digits = datasets.load_digits()
print(digits.data.shape) # 特征空间维度
print(digits.target.shape) # 标签的维度

# 将数据进行分割
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=30)

params = {
    'objective': 'multiclass',
    'num_iterations': 193,
    'num_leaves': 31,
    'learning_rate': 0.1,
}
gbm = LGBMClassifier(**params)

# 训练
gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='multi_logloss', early_stopping_rounds=15)

# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
print(f'Best iterations: {gbm.best_iteration_}')
print(accuracy_score(y_test, y_pred))