# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:52:03 2019

@author: CZW
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing

#数据抓取

def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    
    data = np.array(batch_data)   
    target = np.array(batch_target)
    
    return data, target  

#归一化
def autoNorm(dataSet):
    minVal = dataSet.min()
    maxVal = dataSet.max()
    ranges = maxVal - minVal
    normDateSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    n = dataSet.shape[1]
    normDateSet = dataSet - np.tile(minVal,(m,n))
    normDateSet = normDateSet / np.tile(ranges,(m,n))
    return normDateSet


#转换成 one-hot 格式数据
def oneHotData(data):
    inData = np.array(data)
    outData = np.zeros((inData.shape[0],10))
    for i in range(inData.shape[0]):
        outData[i][data[i]] = 1
    return outData


#10位one-hot数据 转换为数字
def transData(ontHotData):
    ontHotData = ontHotData.reshape(10,)
    ontHotData =ontHotData*10
    ontHotData = ontHotData.astype(np.int32)
    returnData = ontHotData.tolist()
    returnData = returnData.index(max(returnData))
    return returnData

#数据集
mnist = fetch_mldata('MNIST original',data_home=r'E:/CKJ_DeepLearning/mnist')

x = mnist["data"]
y = (mnist["target"]).astype(np.int32)
X_train,X_test, y_train, y_test =train_test_split(x,y,test_size=0.2, random_state=0)


y_train = oneHotData(y_train)
y_test = oneHotData(y_test)

y_test = y_test.astype(np.float64)
y_train = y_train.astype(np.float64)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

X_train = autoNorm(X_train)
X_test = autoNorm(X_test)




#卷积函数---

#权重初始化 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化 
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#实现回归模型 

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

#计算交叉熵
y_ = tf.placeholder("float", [None,10])


#第一层卷积 
  
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1,28,28,1])
x_image = tf.cast(x_image,dtype=tf.float32)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层 
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout 

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy) 

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess: 
    
    sess.run(init)
    
    save_path = saver.save(sess, "my_net/save_net.ckpt")
     
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    batch_xs, batch_ys = next_batch(X_train,y_train,100)
    loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys, keep_prob: 1})
    print(loss)
    

    for i in range(2000):
        batch_xs, batch_ys = next_batch(X_train,y_train,100)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        temp_loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys, keep_prob: 1})
        if i%50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                   x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    
            print("step %d,   accuracy %g, training loss %g"%(i, train_accuracy, temp_loss))
           
    print( "test accuracy %g"%accuracy.eval(feed_dict={
            x: X_test, y_: y_test, keep_prob: 1.0}))
    

    #输出测试图片
    out_xs, out_ys = next_batch(X_test,y_test,20)
    for i in range(19): 
        outNum = (sess.run(y_conv, feed_dict={x: out_xs[i].reshape(1,784), keep_prob: 1.0}))
        num = transData(outNum)
        img = out_xs[i].reshape(28, 28)
        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.show()
        print(num)

