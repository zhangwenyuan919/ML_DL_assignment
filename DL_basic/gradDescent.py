#利用梯度下降算法对一个简单的线性模型进行训练

import tensorflow as tf

# 创建变量 W 和 b 节点，并设置初始值0.1和-0.1
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W*x + b
# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)


# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))#reduce_sum是求和函数


# 创建 Session 用来计算模型
sess = tf.Session()
#初始化变量，否则无法使用变量
init = tf.global_variables_initializer()
sess.run(init)


#创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
x_train = [1,   2,  3,  6,  8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

#训练10000次
for i in range(1,10000):
    #这里train的参数要一直追溯到根，也就是x,y
    sess.run(train, {x: x_train, y: y_train})

#输出训练结果
print("W:%s b:%s loss:%s" % (sess.run(W),sess.run(b),sess.run(loss, {x:x_train, y:y_train})))