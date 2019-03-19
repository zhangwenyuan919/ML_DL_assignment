#使用tf.estimator进行线性回归
import numpy as np
import  tensorflow as tf


#创建一个特征向量表，只有一个特征向量
#该特征向量为实数向量numeric_column，只有一个元素的数组，名称为x
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#创建一个线性回归训练器，并传入特征向量表
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#训练用数据
x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])
#评估用数据
x_eavl = np.array([2., 5., 7., 9.])
y_eavl = np.array([7.6, 17.2, 23.6, 28.8])

# 用训练数据创建一个输入模型，用来进行后面的模型训练
# 第一个参数用来作为线性回归模型的输入数据
# 第二个参数用来作为线性回归模型损失模型的输入
# 第三个参数batch_size表示每批训练数据的个数
# 第四个参数num_epochs为epoch的次数，将训练集的所有数据都训练一遍为1次epoch
# 低五个参数shuffle为取训练数据是顺序取还是随机取
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True
)

# 再用训练数据创建一个输入模型，用来进行后面的模型评估
#此处num_epochs=1000,shuffle=False
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)

# 用评估数据创建一个输入模型，用来进行后面的模型评估
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eavl}, y_eavl, batch_size=2, num_epochs=1000, shuffle=False)

#使用训练数据训练1000次
estimator.train(input_fn=train_input_fn,steps=1000)

#使用原来的训练数据评估模型，目的是查看训练结果
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)

#使用评估数据评估模型，目的是验证模型的泛化性能
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %s" % eval_metrics)