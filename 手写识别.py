# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:37:56 2019

@author: CZW

"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home=r'E:/CKJ_DeepLearning/mnist')

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


x = mnist["data"]
y = (mnist["target"]).astype(np.float64)
X_train,X_test, y_train, y_test =train_test_split(x,y,test_size=0.1, random_state=0)


batch_xs, batch_ys = next_batch(X_train,y_train,10000)
'''
svm_clf = Pipeline((
        ("scaler",StandardScaler()),
        ("linear_svc",LinearSVC(C=1, loss="hinge")),
        ))
svm_clf.fit(batch_xs, batch_ys)
a = svm_clf.predict(batch_xs)
print(svm_clf.score(batch_xs, batch_ys))
'''

clf = RandomForestClassifier(n_estimators = 100,max_leaf_nodes = 100, n_jobs = -1)
clf = AdaBoostClassifier(clf, n_estimators=50, learning_rate=0.2, algorithm='SAMME.R', random_state=None)
clf.fit(batch_xs, batch_ys)


#print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
