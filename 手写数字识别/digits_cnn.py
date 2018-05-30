#coding:utf-8
import os
# 以上为系统库

import pandas as pd
import numpy as np
# 以上为基本库

from sklearn.model_selection import train_test_split # 将数据拆分
from sklearn.metrics import confusion_matrix  # 引入混淆矩阵

from keras.callbacks import ReduceLROnPlateau
# Callbacks提供了一系列的类，用于在训练过程中被回调，从而实现对训练过程进行观察和干涉。
# ReduceLROnPlateau	当指标变化小时，减少学习率

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
# Dense 全连接层 Conv2D 二维卷积层 对输入的二维数据进行卷积操作
# Dropout 失活层 可以避免过拟合
# MaxPool2D 二维池化层
# Flatten Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。



from keras.models import Sequential
# 序列模型是一个线性的层次堆栈。可以通过传递一系列 layer 实例给构造器来创建一个序列模型

from keras.optimizers import RMSprop
# keras 优化器 可以选择的种类有 SGD RMSprop Adagrad Adam Adamax Nadam

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical  # 转换成 one hot encoding

np.random.seed(2) # 产生随机数种子

data_dir = '../../Kaggle_data/digits/'

# load data 
train = pd.read_csv(os.path.join(data_dir,'train.csv'))
test = pd.read_csv(os.path.join(data_dir,'test.csv')) 

X_train = train.values[:, 1:]
Y_train = train.values[:, 0]

test = test.values

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes=10)\

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)


# 下面开始搭建 CNN

model = Sequential()

# 卷积层
model.add(
    Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1) )
)

model.add(
    Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu')
)

# 池化层
model.add(
    MaxPool2D(pool_size=(2,2))
)

# dropout
model.add(
    Dropout(0.25)
)

model.add(
    Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')
)

model.add(
    Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu')
)

# 池化层
model.add(
    MaxPool2D(pool_size=(2,2), strides=(2, 2))
)

# dropout
model.add(
    Dropout(0.25)
)

model.add(
    Flatten()
)

# 全连接层
model.add(
    Dense(256, activation='relu')
)

# dropout
model.add(
    Dropout(0.5)
)

# 连接到输出层
model.add(
    Dense(10, activation='softmax')
)

# 定义优化器
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 30
batch_size = 86

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001
)
# 当评价指标不在提升时，减少学习率
# factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
# min_lr：学习率的下限
# monitor：被监测的量
# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作


# 对数据进行扩充
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction])

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat(
    [pd.Series(
        range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv(os.path.join(data_dir, "../手写数字识别/Result_keras_CNN.csv",index=False))
print('finished')