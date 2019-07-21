# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:54:30 2019

@author: CZW
"""


import numpy as np  
import matplotlib.pyplot as plt  
    

#参数设置
train_step = 5000
learning_rate = 0.00005


#生成数据
#x的个数决定了样本量
x = np.arange(-2,10,0.02)
#y为理想函数
k_ = [1000,2] 
y = np.ones(x.shape)

#for i in range(len(k_)):
#    a = k_[i]*x**(i)
#    y = a+y
y = 5*x + 2000

#y1为离散的拟合数据
y1 = y+0.6*(np.random.rand(len(x))-0.5) 



##################################
#主要程序
#系数
k = np.array([-2,5])

def iteration(x,y1,k,learning_rate,train_step):  
    # y1:单列真实值； y:估值子集； y_：多列
    loss = np.array([10,10])
    step = 0
    x_ = np.ones(x.shape)
    for i in range(len(k)-1):
        x_ = np.c_[x_,x**(i+1)]
    y = np.ones(x_.shape)
    #训练系数项
    while np.dot(np.ones(loss.shape),abs(loss)) > 1e-5 and step < 3000:
        step += 1
        y = np.dot(k,x_.T)
        J = (y-y1)
        loss = np.dot(x_.T,J)*2
        k = k - learning_rate*loss
#        print(loss,step,k)
  
    print(k)     
    yy = np.dot(k,x_.T)
    return yy
yy = iteration(x,y1,k,learning_rate,train_step)
##################################
#绘制图像
#plt.plot(x,y,color='g',linestyle='-',marker='',label=u'理想曲线') 
plt.plot(x,y1,color='y',linestyle=' ',marker='o',label=u'拟合数据')
plt.plot(x,yy,color='b',linestyle='-',marker='.',label=u"拟合曲线") 
# 把拟合的曲线在这里画出来
plt.legend(loc='upper left')
plt.show()
