# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:31:05 2019

@author: CZW
"""

from sklearn import linear_model
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import  pandas  as pd
from sklearn.linear_model import LinearRegression
import random
import xlsxwriter as xlwt

#随机数生成
X_data = []
E_data = []
Y_temp = []
Y_data = []
for i in range(120):
    tempX = random.randint(0,1500)
    X_data.append(tempX)
    tempE = random.randint(-700,700)
    E_data.append(tempE)

for i in range(120): 
    tempY = 4000 + 5*X_data[i] + E_data[i] 
    Y_temp.append(tempY)  
    tempy = 4000 + 5*X_data[i] 
    Y_data.append(tempy)  
    
#直线拟合

x0Data = np.array(X_data).reshape(120,1)
y1Data = np.array(Y_temp).reshape(120,1)
y0Data = np.array(Y_data).reshape(120,1)


clf = linear_model.LinearRegression()
clf.fit(x0Data,y1Data)

print(clf.coef_)
print(clf.intercept_)














