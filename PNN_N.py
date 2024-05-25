# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:28:08 2022

http://neupy.com/docs/tutorials.html
@author: https://github.com/itdxer/neupy/blob/master/examples/rbfn/pnn_iris.py

"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from neupy.algorithms import PNN
import hickle as hkl

x,y_t,x_norm,x_n_s,y_t_s = hkl.load('iris.hkl')
y_t -= 1
x=x.T
y_t = np.squeeze(y_t)

data = x
target = y_t

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)

for i, (train, test) in enumerate(skfold.split(data, target), start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = target[train], target[test]
    # print(i,train,test)
    pnn_network = PNN(std=0.1, verbose=False)
    pnn_network.train(x_train, y_train)
    result = pnn_network.predict(x_test)

    n_test_samples = test.size
    PK_vec[i] = np.sum(result == y_test) / n_test_samples
    
    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))

PK = np.mean(PK_vec)
print("PK {}".format(PK))