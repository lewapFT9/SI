import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
        
class mlp_ma_3w:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, ksi_inc, ksi_dec, er, mc, max_epoch):
        self.x          = x
        self.L          = self.x.shape[0] 
        self.y_t        = y_t
        self.K1         = K1
        self.K2         = K2
        self.lr         = lr
        self.err_goal   = err_goal
        self.ksi_inc    = ksi_inc
        self.ksi_dec    = ksi_dec
        self.er         = er
        self.mc         = mc
        self.max_epoch  = max_epoch
        self.K3         = y_t.shape[0]
        self.SSE_vec    = [] 
        self.PK_vec     = []
        self.data       = self.x.T
        self.target     = self.y_t
    
        self.w1, self.b1 = net.nwtan(self.K1, self.L)  
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        
        self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
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
            self.e = y_train - self.y3 
        
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
            self.dw1, self.db1 = net.learnbp(x_train,  self.d1, self.lr) 
            self.dw2, self.db2 = net.learnbp(self.y1,  self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2,  self.d3, self.lr)
            
            self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp = \
                self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(), self.w3.copy(), self.b3.copy()
            
            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1) 
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1) 
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1) 
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1) 
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1) 
            
            self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 =  \
                self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp
            
            self.SSE_vec.append(self.SSE) 
            
    def train_CV(self, CV, skfold):
        PK_vec = np.zeros(CVN)
        
        for i, (train, test) in enumerate(skfold.split(self.data, np.squeeze(self.target)), start=0): #sqeeze removes 1-dimensional subsets form array

            x_train, x_test = self.data[train], self.data[test]
            y_train, y_test = np.squeeze(self.target)[train], np.squeeze(self.target)[test]
            
            self.train(x_train.T, y_train.T)
            result = self.predict(x_test.T)
        
            n_test_samples = test.size
            PK_vec[i] = (1 - sum((abs(result - y_test)>=0.5).astype(int)[0])/n_test_samples ) * 100
            
        PK = np.mean(PK_vec)
        return PK


x,y_t,x_norm,x_n_s,y_t_s = hkl.load('norm_red_wine_2.hkl')

max_epoch = 7500
err_goal = 0.25
lr = 1e-4
ksi_inc = np.arange(1.01,1.41,0.025)               #learning rate increase
ksi_dec = np.arange(0.5,0.99,0.1)                  #learinig rate decrease
er_vec = np.arange(1.01,1.52,0.05)                 #error rate
mc_vec = np.array([0.01,0.1,0.3,0.5,0.7,0.9,0.99]) #momentum constans
K1_vec = np.arange(3,32,3)
K2_vec = K1_vec

start = timer()

CVN = 10

skfold = StratifiedKFold(n_splits=CVN)
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])
PK_vec = np.zeros(CVN)
PK_2D_ermc = np.zeros([len(er_vec),len(mc_vec)])
PK_2D_K1K2_max = 0
PK_2D_ksi=  np.zeros([len(ksi_inc),len(ksi_dec)])
k1_ind_max = 0 
k2_ind_max = 0
for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        mlpnet = mlp_ma_3w(x_norm, y_t, K1_vec[k1_ind], K2_vec[k2_ind], lr, err_goal, ksi_inc[0],ksi_dec[-1],er_vec[0], mc_vec[-1], max_epoch)
        PK = mlpnet.train_CV(CVN, skfold)
        print("K1 {} | K2 {} | PK {}". format(K1_vec[k1_ind], K2_vec[k2_ind], PK))
        PK_2D_K1K2[k1_ind, k2_ind] = PK
        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind 
            k2_ind_max = k2_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')

ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_parkinsons.png",bbox_inches='tight')

PK_2D_ermc_max = 0
er_ind_max = 0 
mc_ind_max = 0

for er_ind in range(len(er_vec)):
    for mc_ind in range(len(mc_vec)):
        mlpnet = mlp_ma_3w(x_norm, y_t, K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr, err_goal, ksi_inc[0],ksi_dec[-1],er_vec[er_ind], mc_vec[mc_ind], max_epoch)
        PK = mlpnet.train_CV(CVN, skfold)
        print("er {} | mc {} | PK {}".format(er_vec[er_ind], mc_vec[mc_ind], PK))
        PK_2D_ermc[er_ind, mc_ind] = PK
        if PK > PK_2D_ermc_max:
            PK_2D_ermc_max = PK
            er_ind_max = er_ind 
            mc_ind_max = mc_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(er_vec, mc_vec)
surf = ax.plot_surface(X, Y, PK_2D_ermc.T, cmap='viridis')

ax.set_xlabel('er')
ax.set_ylabel('mc')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.2_PK_ermc_parkinsons.png",bbox_inches='tight')


PK_2D_ksi_max = 0
ksi_inc_ind_max = 0 
ksi_dec_ind_max = 0

for ksi_inc_ind in range(len(ksi_inc)):
    for ksi_dec_ind in range(len(ksi_dec)):
        mlpnet = mlp_ma_3w(x_norm, y_t, K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr, err_goal, \
                           ksi_inc[ksi_inc_ind],ksi_dec[ksi_dec_ind],er_vec[er_ind_max], mc_vec[mc_ind_max], max_epoch)
        PK = mlpnet.train_CV(CVN, skfold)
        print("ksi_inc {} | ksi_dec {} | PK {}".format(ksi_inc[ksi_inc_ind], ksi_dec[ksi_dec_ind], PK))
        PK_2D_ksi[ksi_inc_ind, ksi_dec_ind] = PK
        if PK > PK_2D_ksi_max:
            PK_2D_ksi_max = PK
            ksi_inc_ind_max = ksi_inc_ind
            ksi_dec_ind_max = ksi_dec_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(ksi_inc, ksi_dec)
surf = ax.plot_surface(X, Y, PK_2D_ksi.T, cmap='viridis')

ax.set_xlabel('ksi_inc')
ax.set_ylabel('ksi_dec')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.3_PK_ksi_parkinsons.png",bbox_inches='tight')
plt.show()            
print("OPTYMALNE WARTOŚCI: K1={} | K2={} | er={} | mc={} | ksi_inc={} | ksi_dec={} | PK={}".\
      format(K1_vec[k1_ind_max], K2_vec[k2_ind_max], er_vec[er_ind_max], mc_vec[mc_ind_max],\
             ksi_inc[ksi_inc_ind_max],ksi_dec[ksi_dec_ind_max], PK_2D_ermc[er_ind_max, mc_ind_max]))            

print("Execution time:", timer()-start)


#OPTYMALNE WARTOŚCI: K1=10 | K2=7 | er=1.01 | mc=0.12999999999999998 | ksi_inc=1.2100000000000002 | ksi_dec=0.91 | PK=99.5
#OPTYMALNE WARTOŚCI: K1=10 | K2=31 | er=1.2100000000000002 | mc=0.91 | ksi_inc=1.9100000000000008 | ksi_dec=0.91 | PK=99.0

