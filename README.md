# Kaggle

kaggle -- 房价预测
       -- 手写数据识别

### 解决问题的流程
1. 链接场景和目标
2. 链接评估准则
3. 认识数据
4. 数据预处理（数据清洗、调权）
5. 特征工程
6. 模型调参
7. 模型状态分析
8. 模型融合

详细介绍：  

* 数据清洗
    (1) 去掉样本数据的异常数据（比如连续数据中的离散点）
    (2) 去除缺失大量特征的数据
* 数据采样
    (1) 上\下采样
    目的：保证样本的均衡
* 特征工程
数据预处理：  

    标准化：零均值单位方差scale
    区间缩放：最大最小值标准化、绝对值最大标准化、对稀疏数据进行标准化、对离群点标准化
    正则化：范数、Normalizer
    二值化：特征二值化
    对类别特征进行编码：one_hot_encoder
    缺失值计算：
    生成多项式特征：
    自定义转换：

 特征提取：
    从字典类型加载特征：
    特征散列：
    文本特征提取：
    图像特征提取：

特征选择：
    Filter：方差选择法、相关系数法、卡方检验、互信息法
    Wrapper：递归特征消除法
    Embedded：基于惩罚项的特征选择法、基于树模型的特征选择法

降维：
    主成分分析法（PCA）
    线性判别分析（LDA）

使用 vscode 进行可视化操作

## 第一次（初始化）
clone 自己的repo
git clone https://github.com/super1peng/Kaggle.git

## 进入该仓库文件夹
cd kaggle

## 查看该仓库远程repo
git remote

## 添加 远程 repo 仓库（添加一次之后就不用再进行添加了）
git remote add origin_online https://github.com/super1peng/Kaggle.git

## 第二次（修改并进行提交）
git pull origin_online matser

## 上传到自己的repo仓库
git push origin master