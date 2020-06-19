import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.utils import to_categorical
from  keras.metrics import categorical_crossentropy
from keras.optimizers import Adam


# 导入mnist数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 显示一下数据是否正确
# 首先把训练集和测试集都转换成28*28*1的灰白图片矩阵的形式，方便进行卷积操作
# 整体做归一化处理
x_train = x_train.reshape(len(x_train), 28, 28, 1) / 255
x_test = x_test.reshape(len(x_test), 28, 28, 1) / 255

# 事实证明 mnist 的labels 是1，2，3这种的，需要转成one-hot向量
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


'''
测试一下图片能否显示
show_img = np.array(x_train[0], dtype=float).reshape((28, 28))
plt.imshow(show_img)
plt.show()
'''

# 定义一个序列模型
model = Sequential()
# 首先加入第一个卷积层 6组5x5的Filter
# 得到6组 24x24x1的特征映射
# 本层的输入既mnist数据集  28x28
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
# 加入池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# 再次加入卷积层 本次Filter大小仍然为5x5
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# 加入池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# 接入一个FCNN,这里接入FCNN的原因就是，在本层的输入是4x4x16 而kernel_size为5x5，已经不能再进行一次卷积了
# 但是具体如何还是要看一眼paper， 我只是个菜鸡，就不在这里丢人现眼了QAQ
# 首先将整个数据展开
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
# 最后一层定义为输出层
model.add(Dense(10, activation='softmax'))


# 最后进行编译以及预测 使用交叉熵作为损失函数
# 使用Rmsprop优化器
model.compile(loss=categorical_crossentropy, optimizer='Rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Final Loss: %f and Final accuracy: %f' % (score[0], score[1]))


