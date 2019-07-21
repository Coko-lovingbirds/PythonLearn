# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:33:16 2019

@author: CZW  线性回归
"""

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
    


x0Data = np.array(X_data).reshape(120,1)
y1Data = np.array(Y_temp).reshape(120,1)
y0Data = np.array(Y_data).reshape(120,1)


#输出到excel
data=np.hstack((x0Data,y1Data))

data_df = pd.DataFrame(data, columns=['X_data','Y_data'])
writer = pd.ExcelWriter(r'C:\Users\CZW\Desktop\机器学习课\data.xlsx')

# create and writer pd.DataFrame to excel
data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()



#读excel数据
def readExcel(m):   
    df = pd.read_excel(r'C:\Users\CZW\Desktop\机器学习课\data.xlsx', usecols=[m],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result
X_data = readExcel(1)
Y_temp = readExcel(2)
x0Data = np.array(X_data).reshape(120,1)
y1Data = np.array(Y_temp).reshape(120,1)


one=np.ones((120,1))
data=np.hstack((x0Data,one))#两个100x1列向量合并成100x2,(100, 1) (100,1 ) (100, 2)
print(len(x0Data))

def optimal(A,b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))#求A.T.dot(A)的逆
    P=AA.dot(B)
    print(P)
    return A.dot(P)

xy=optimal(data,y1Data)

#绘图 
#plt.plot(x0Data,y0Data,color='g',linestyle='-',marker='',label=u'理想曲线') 
plt.plot(x0Data,y1Data,color='m',linestyle='',marker='o',label=u'拟合数据')
plt.plot(x0Data,xy,color='b',linestyle='-',marker='.',label=u"拟合曲线") 
# 把拟合的曲线在这里画出来
plt.legend(loc='upper left')
plt.show()

