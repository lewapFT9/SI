import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt

x = np.array([[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5]])
y_t = np.array([[-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600]])

max_epoch = 20000
err_goal = 1e-10
disp_freq = 1000
lr = 0.01
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04
L = x.shape[0]
K1 = 3
K2 = 4
K3 = y_t.shape[0]
print(K3)
SSE_vec = []
# w1, b1 = net.nwtan(K1, L)
# w2, b2 = net.nwtan(K2, K1)
# w3, b3 = net.rands(K3, K2)
# hkl.dump([w1, b1, w2, b2, w3, b3], 'wagi3w_a.hkl')
w1, b1, w2, b2, w3, b3 = hkl.load('wagi3w_a.hkl')

SSE = 0
lr_vec = list()

for epoch in range(1, max_epoch + 1):
    y1 = net.tansig(np.dot(w1, x), b1)
    y2 = net.tansig(np.dot(w2, y1), b2)
    y3 = net.purelin(np.dot(w3, y2), b3)
    e = y_t - y3

    SSE_t_1 = SSE
    SSE = net.sumsqr(e)
    if np.isnan(SSE):
        break
    else:
        if SSE > er * SSE_t_1:
            lr *= ksi_dec
        elif SSE < SSE_t_1:
            lr *= ksi_inc
    lr_vec.append(lr)

    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3)
    d1 = net.deltatan(y1, d2, w2)
    dw1, db1 = net.learnbp(x, d1, lr)
    dw2, db2 = net.learnbp(y1, d2, lr)
    dw3, db3 = net.learnbp(y2, d3, lr)

    w1 += dw1
    b1 += db1
    w2 += dw2
    b2 += db2
    w3 += dw3
    b3 += db3

    SSE_vec.append(SSE)

    if SSE < err_goal:
        break
    if (epoch % disp_freq) == 0:
        print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))

        # plt.clf()
        # plt.plot(x[0],y_t[0],'r',x[0],y3[0],'g')
        # plt.grid()
        # plt.show()
        # plt.pause(1e-2)

print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
hkl.dump([SSE_vec], 'SSE2w_adapt.hkl')

plt.plot(SSE_vec)
plt.ylabel('SSE')
plt.yscale('linear')
plt.title('epoch')
plt.grid(True)
plt.show()
