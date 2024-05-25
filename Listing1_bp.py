
import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt 

#x   = np.array([[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5]])
#y_t = np.array([[-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600]])

#x, y_t = hkl.load("norm_red_wine.hkl")
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('iris.hkl')


max_epoch = 20000
err_goal = 0.1
disp_freq = 400
lr = 1e-4
L = x_norm.shape[0]
K1 = 7
K2 = 3
K3 = 1
SSE_vec = [] 
w1, b1 = net.nwtan(K1, L)
w2, b2 = net.nwtan(K2, K1)
w3, b3 = net.rands(K3,K2)
hkl.dump([w1,b1,w2,b2,w3,b3], 'wagi2w.hkl')
w1,b1,w2,b2,w3,b3 = hkl.load('wagi2w.hkl')

#print(x.shape[0],x.shape[1])
for epoch in range(1, max_epoch+1): 
    y1 = net.tansig(np.dot(w1, x_norm), b1)
    y2 = net.tansig(np.dot(w2, y1), b2) #
    y3 = net.purelin(np.dot(w3, y2), b3)
    e = y_t - y3
    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3)
    d1 = net.deltatan(y1, d2, w2) 
    dw1, db1 = net.learnbp(x_norm,  d1, lr)
    dw2, db2 = net.learnbp(y1, d2, lr)
    dw3, db3 = net.learnbp(y2, d3, lr)
    w1 += dw1 
    b1 += db1 
    w2 += dw2 
    b2 += db2
    w3 += dw3
    b3 += db3
 
    SSE = net.sumsqr(e) 
    if np.isnan(SSE): 
        break

    SSE_vec.append(SSE) 
 
    if SSE < err_goal: 
        break 
    if (epoch % disp_freq) == 0:
        print(SSE)
        print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
        plt.clf()
        plt.plot(x_norm[0],y_t[0],'r',x[0],y3[0],'g')
        plt.grid()
        plt.show()
        plt.pause(1e-2)

print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
hkl.dump([SSE_vec], 'SSE2w_gradient.hkl')
        
plt.plot(SSE_vec)
plt.ylabel('SSE')
plt.yscale('linear')
plt.title('epoch')
plt.grid(True)
plt.show()
