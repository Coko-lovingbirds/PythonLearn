# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:33:50 2019

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
    
    a = np.array(batch_data)   
    b = np.array(batch_target)
    return a, b  


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

#转换成 one-hot 格式数据
def oneHotData(data):
    inData = np.array(data)
    outData = np.zeros((inData.shape[0],10))
    for i in range(inData.shape[0]):
        outData[i][data[i]] = 1
    return outData

y_train = oneHotData(y_train)
y_test = oneHotData(y_test)

X_train = autoNorm(X_train)
X_test = autoNorm(X_test)


#实现回归模型 

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

#计算交叉熵
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#梯度下降算法(gradient descent algorithm)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy) 


init = tf.initialize_all_variables()

#运行
with tf.Session() as sess: 
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    for i in range(2000):
        batch_xs, batch_ys = next_batch(X_train,y_train,100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        temp_loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})
        if i%200 == 0:    
            print("step %d, training loss %g"%(i, temp_loss))
    
        
#    print(sess.run(y, feed_dict={x: X_train[10000].reshape(1,784)}))
    
   
#    print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    print(sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))


#输出测试图片
    out_xs, out_ys = next_batch(X_test,y_test,10)
    for i in range(9):
        outNum = (sess.run(y, feed_dict={x: out_xs[i].reshape(1,784)}))
        num = transData(outNum)
        img = out_xs[i].reshape(28, 28)
        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.show()
        print(num)


