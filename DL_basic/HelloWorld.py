import tensorflow as tf


############################
#初识TensorFlow
hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))




##############################
#关于张量Tensor
t0 = tf.constant(3, dtype=tf.int32)

#1阶张量，即1维数组
t1 = tf.constant([3., 4.1 , 5.2], dtype=tf.float32)

#2阶Tensor,2*2
t2 = tf.constant([['Apple' , 'Orange'] , ['Potato' , 'Tomato']],dtype=tf.string)

#3阶Tensor，2*3*1
t3 = tf.constant([[[5],[6],[7]],[[4],[3],[2]]])

print(t0)
print(t1)
print(t2)
print(t3)


print(sess.run(t0))
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))



#############################
#对张量的操作，数据流图
#创建两个常量节点
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
#创建一个adder节点，对上面两个节点执行+操作
adder = node1 + node2

print(adder)
print(sess.run(adder))



###############################
#使用外部输入变量，而非常量
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
print(a)
print(b)
print(adder_node)

print(sess.run(adder_node , {a: 3, b: 4.5}))
print(sess.run(adder_node , {a:[1,3], b:[2,4]}))








