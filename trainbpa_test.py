import hickle as hkl
import numpy as np
#import nnet as net
import nnet_jit4 as net
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.ticker as ticker


class BPA:
    def __init__(self, X, y, K1, K2, lr, error_goal, ksi_inc, ksi_dec, er, max_epoch):
        self.X = X
        self.y = y
        self.K1 = K1
        self.K2 = K2
        self.lr = lr
        self.error_goal = error_goal
        self.ksi_inc = ksi_inc
        self.ksi_dec = ksi_dec
        self.er = er
        self.max_epoch = max_epoch
        self.L = X.shape[0]
        self.K3 = y.shape[0]

        self.SSE_VEC = []
        self.PK_VEC = []
        self.accuracy = []
        self.SSE = 0
        self.lr_vec = list()
        self.disp_f = 10

        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)


    def predict(self, X):
        self.y1 = net.tansig(np.dot(self.w1, X.T), self.b1)
        self.y2 = net.tansig(np.dot(self.w2, self.y1), self.b2)
        self.y3 = net.purelin(np.dot(self.w3, self.y2), self.b3)
        return self.y3


    def train(self, X_train, y_train, X_test, y_test):
        for epoch in range(self.max_epoch):
            self.y3 = self.predict(X_train)
            self.e = y_train.T - self.y3
            self.SSE_tm1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (1 - sum((abs(self.e)>=0.5).astype(int)[0])/self.e.shape[1] ) * 100
            self.PK_VEC.append(self.PK)

            if self.SSE < self.error_goal or self.PK == 100:
                break

            if np.isnan(self.SSE):
                break
            if self.SSE > self.er * self.SSE_tm1:
                self.lr *= self.ksi_dec
            elif self.SSE < self.SSE_tm1:
                self.lr *= self.ksi_inc
            self.lr_vec.append(self.lr)

            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(X_train.T, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1 += self.dw1
            self.b1 += self.db1
            self.w2 += self.dw2
            self.b2 += self.db2
            self.w3 += self.dw3
            self.b3 += self.db3

            self.SSE_VEC.append(self.SSE)

            self.score = self.predict(X_test)
            #if epoch % self.disp_f == 0:
            self.accuracy.append((sum((abs(self.score - y_test)<0.5).astype(int)[0])/X_test.shape[0]) * 100)

        return np.mean(self.accuracy)




x, y_t, x_norm, x_n_s, y_t_s = hkl.load('norm_white_wine.hkl')
data = x_norm.T
target = y_t.T
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=46)


max_epoch = 2000
err_goal = 100
disp_freq = 100
lr = 1e-4
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04

K1_vec = np.arange(3,32,3)
K2_vec = K1_vec
ksi_inc_vec = np.arange(1.01,1.30,0.02)
ksi_dec_vec = np.arange(0.5,0.9,0.05)
PK_inc_dec = np.zeros([len(ksi_inc_vec),len(ksi_dec_vec)])
PK_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])
k1_ind_max = 0
k2_ind_max = 0
PK_K1K2_max = 0

PK_INC_DEC_MAX = 0
ksi_dec_opt = 0
ksi_inc_opt = 0
PK_INC_DEC = np.zeros([len(ksi_inc_vec), len(ksi_dec_vec)])

PK_LR_MAX = 0
lr_opt = 0
lr_vec = [0.000001, 0.000002, 0.000005, 0.000007, 0.000009, 0.00001, 0.00002,
          0.00005, 0.00007, 0.000085, 0.0001,
          0.00012, 0.00015]
PK_LR = np.zeros(len(lr_vec))

data = x_norm
target = y_t


for k1 in range(len(K1_vec)):
    for k2 in range(len(K2_vec)):

        bpa_nn_1 = BPA(data, target, K1_vec[k1], K2_vec[k2], lr, err_goal,
                       ksi_inc, ksi_dec, er, max_epoch)
        PK = bpa_nn_1.train(X_train, y_train, X_test, y_test)
        PK_K1K2[k1][k2] = PK
        print(f"K1: {K1_vec[k1]}; K2: {K2_vec[k2]}; PK: {PK}")
        if PK > PK_K1K2_max:
            PK_K1K2_max = PK
            k1_ind_max = K1_vec[k1]
            k2_ind_max = K2_vec[k2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x_w, y_w = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(x_w, y_w, PK_K1K2.T, cmap='viridis')

ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_WINE_QUALITY_Red.png", bbox_inches='tight')
for inc in range(len(ksi_inc_vec)):
    for dec in range(len(ksi_dec_vec)):
        bpa_nn_2 = BPA(data,target,k1_ind_max, k2_ind_max, lr, err_goal,
                       ksi_inc_vec[inc], ksi_dec_vec[dec], er,
                       max_epoch)
        PK = bpa_nn_2.train(X_train, y_train, X_test, y_test)
        PK_inc_dec[inc][dec] = PK
        print(f"inc: {ksi_inc_vec[inc]}, dec: {ksi_dec_vec[dec]}, PK: {PK}")
        if PK > PK_INC_DEC_MAX:
            PK_INC_DEC_MAX = PK
            ksi_inc_opt = ksi_inc_vec[inc]
            ksi_dec_opt = ksi_dec_vec[dec]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x_w, y_w = np.meshgrid(ksi_inc_vec, ksi_dec_vec)
surf = ax.plot_surface(x_w, y_w, PK_inc_dec.T, cmap='viridis')

ax.set_xlabel('ksi_inc')
ax.set_ylabel('ksi_dec')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.2_PK_inc_dec_WINE_QUALITY_Red.png", bbox_inches='tight')


for lr in range(len(lr_vec)):
    bpa_nn_3 = BPA(data, target, k1_ind_max,k2_ind_max,lr_vec[lr],err_goal,
                   ksi_inc_opt,ksi_dec_opt,er,max_epoch)
    PK = bpa_nn_3.train(X_train, y_train, X_test, y_test)
    PK_LR[lr] = PK
    print(f"lr: {lr_vec[lr]}, PK: {PK}")
    if PK > PK_LR_MAX:
        PK_LR_MAX = PK
        lr_opt = lr_vec[lr]

fig = plt.figure()
plt.plot(lr_vec,PK_LR)
plt.xlabel("Learning Rate")
plt.ylabel("PK")
plt.grid(True)
plt.show()
fig.savefig("Fig.3_PK_LR_Red.png", bbox_inches='tight')


bpa_nn_4 = BPA(data,target, k1_ind_max, k2_ind_max, lr_opt, err_goal,
               ksi_inc_opt, ksi_dec_opt,
               er,max_epoch)
PK = bpa_nn_4.train(X_train, y_train, X_test, y_test)
SSE_VEC = bpa_nn_4.SSE_VEC
PK_vec = bpa_nn_4.accuracy
lr_VEC = bpa_nn_4.lr_vec
print(f"WYNIK DLA OPTYMALNYCH PARAMETRÓW: K1: {k1_ind_max}, K2: {k2_ind_max},"
      f" LR: {lr_opt},KSI_INC: {ksi_inc_opt}, "
      f"KSI_DEC: {ksi_dec_opt}, PK: {PK}")

fig = plt.figure()
plt.plot(lr_VEC)
plt.title("lr = f(epoch)")
plt.xlabel("epoch")
plt.ylabel("lr")
plt.grid(True)
plt.show()
fig.savefig("Fig.7_opt_lr.png", bbox_inches='tight')

fig = plt.figure()
plt.plot(SSE_VEC)
plt.title("SSE = f(epoch)")
plt.xlabel("epoch")
plt.ylabel("SSE")
plt.grid(True)
plt.show()
fig.savefig("Fig.8_opt_SSE.png", bbox_inches='tight')

fig = plt.figure()
plt.plot(PK_vec)
plt.title("PK = f(epoch)")
plt.xlabel("epoch")
plt.ylabel("PK")
plt.grid(True)
plt.show()
fig.savefig("Fig.9_opt_PK.png", bbox_inches='tight')








