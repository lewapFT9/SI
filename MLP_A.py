# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:23:50 2022

@author: RZ
"""

import hickle as hkl
import numpy as np
#import nnet as net
import nnet_jit4 as net
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
        
class mlp_a_3w:
    def __init__(self, x, y_t, K1, K2, K3, lr, err_goal, \
                 disp_freq, ksi_inc, ksi_dec, er, max_epoch):
        self.x          = x
        self.L          = self.x.shape[1] 
        self.y_t        = y_t
        self.K1         = K1
        self.K2         = K2
        self.K3         = K3
        self.lr         = lr
        self.err_goal   = err_goal
        self.disp_freq  = disp_freq
        self.ksi_inc    = ksi_inc
        self.ksi_dec    = ksi_dec
        self.er         = er
        self.max_epoch  = max_epoch

        self.SSE_vec = [] 
        self.PK_vec = []

        self.w1, self.b1 = net.nwtan(self.K1, self.L)  
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        # hkl.dump([self.w1,self.b1,self.w2,self.b2,self.w3,self.b3], 'wagi3w.hkl')
        # self.w1,self.b1,self.w2,self.b2,self.w3,self.b3 = hkl.load('wagi3w.hkl')
        self.SSE = 0
        self.lr_vec = list()
    
    def predict(self,x):
        n = np.dot(self.w1, x)
        self.y1 = net.tansig( n,  self.b1*np.ones(n.shape)) 
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig( n,  self.b2*np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.purelin(n, self.b3*np.ones(n.shape)) 
        return self.y3
        
    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch+1): 
            self.y3 = self.predict(x_train)    
            self.e = y_train.T - self.y3
        
            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e) 
            self.PK = (1 - sum((abs(self.e)>=0.5).astype(int)[0])/self.e.shape[1] ) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100: 
                break 
            
            if np.isnan(self.SSE): 
                break
            else:
                if self.SSE > self.er * self.SSE_t_1:
                    self.lr *= self.ksi_dec
                elif self.SSE < self.SSE_t_1:
                    self.lr *= self.ksi_inc
            self.lr_vec.append(self.lr)
            
            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2) 
            self.dw1, self.db1 = net.learnbp(self.x.T,  self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)
            
            self.w1 += self.dw1 
            self.b1 += self.db1  
            self.w2 += self.dw2  
            self.b2 += self.db2 
            self.w3 += self.dw3  
            self.b3 += self.db3  
            
            self.SSE_vec.append(self.SSE) 
            

x,y_t,x_norm,x_n_s,y_t_s = hkl.load('iris.hkl')

max_epoch = 2000
err_goal = 0.01
disp_freq = 10 
lr = 1e-4
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04
K1 = 6
K2 = 3

data = x_n_s.T
target = y_t_s
K3 = target.shape[0] if len(target.shape)>1 else 1

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)

for i, (train, test) in enumerate(skfold.split(data, np.squeeze(target)), start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]
    
    mlpnet = mlp_a_3w(x_train, y_train, K1, K2, K3, lr, err_goal, \
                     disp_freq, ksi_inc, ksi_dec, er,max_epoch)
    mlpnet.train(x_train.T, y_train.T)
    result = mlpnet.predict(x_test.T)

    n_test_samples = test.size
    PK_vec[i] = sum((abs(result - y_test)<0.5).astype(int)[0])/n_test_samples * 100
    print(mlpnet.SSE)
    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))

PK = np.mean(PK_vec)
print("PK {}".format(PK))
plt.figure()
plt.plot(mlpnet.lr_vec)
plt.title("lr=f(epoch)")
plt.figure()
plt.plot(mlpnet.SSE_vec)
plt.title("SSE=f(epoch)")
plt.show()

print(len(PK_vec))