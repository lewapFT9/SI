import hickle as hkl
import numpy as np
#import nnet as net
import nnet_jit4 as net
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd



x, y_t, x_norm, x_n_s, y_t_s = hkl.load('norm_white_wine.hkl')



max_epoch = 2000
err_goal = 2
disp_freq = 100
lr = 0.001
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04
K1 = 8
K2 = 4

data = x_norm
target = y_t
K3 = target.shape[0]
SSE_VEC=[]
lr_VEC = []

w1, b1 = net.nwtan(K1, data.shape[0])
w2, b2 = net.nwtan(K2, K1)
w3, b3 = net.rands(K3, K2)
hkl.dump([w1,b1,w2,b2,w3,b3], 'wagi2wq.hkl')
# w1,b1,w2,b2,w3,b3 = hkl.load('wagi2wq.hkl')

# print(w1)
# print(w1.shape[0],w1.shape[1])

for epoch in range(max_epoch):
    y1 = net.tansig(np.dot(w1,data), b1)
    y2 = net.tansig(np.dot(w2,y1), b2)
    y3 = net.purelin(np.dot(w3,y2), b3)
    e = target - y3
    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3)
    d1 = net.deltatan(y1, d2, w2)

    dw1, db1 = net.learnbp(data,d1,lr)
    dw2, db2 = net.learnbp(y1,d2,lr)
    dw3, db3 = net.learnbp(y2,d3,lr)

    w1+=dw1
    b1+=db1
    w2+=dw2
    b2+=db2
    w3+=dw3
    b3+=db3

    SSE= net.sumsqr(e)
    if np.isnan(SSE):
        break;
    SSE_VEC.append(SSE)

    if SSE < err_goal:
        break

    if SSE > er * SSE_VEC[epoch-1]:
        lr*=ksi_dec
    elif SSE < er * SSE_VEC[epoch-1]:
        lr*=ksi_inc
    lr_VEC.append(lr)

    if epoch % disp_freq == 0:
        print('Epoch:',epoch)
        print('SSE:',SSE)
        print("Correct: ",sum((abs(y3-target) < 0.5).astype(int)[0]) / data.shape[1] * 100,"%")
        # print("predict: ", y3[0],"\n", "target: ", target[0] )

plt.plot(SSE_VEC)
plt.ylabel('SSE')
#plt.yscale('linear')
plt.grid(True)
plt.show()

plt.plot(lr_VEC)
plt.ylabel('lr')
plt.yscale('linear')
plt.grid(True)
plt.show()

