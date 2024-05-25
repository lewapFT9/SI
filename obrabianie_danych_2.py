import hickle as hkl
import numpy as np

filename = "czerwone_wina_t.txt"
data = np.loadtxt(filename,delimiter=",", dtype=str)
licznik = 1
data = np.delete(data,0,axis=0)

for i in range(0,len(data)+1):
    if i==len(data)-licznik:
        break
    for j in range(0,len(data[i])):
        if data[i][j] == "":
            licznik+=1
            data = np.delete(data, i, axis=0)
            if data[i][j] == "":
                data = np.delete(data, i, axis=0)

x = data[:, 0:-1].astype(float).T
y_t = data[:, -1].astype(int)
y_t = y_t.reshape(1,y_t.shape[0])

x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)

for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
        (x[i,:]-x_min[i]) + x_norm_min

y_t_s_ind = np.argsort(y_t)
x_n_s = np.zeros(x.shape)
y_t_s = np.zeros(y_t.shape)
for i in range(x.shape[1]):
    y_t_s[0,i] = y_t[0,y_t_s_ind[0,i]]
    x_n_s[:,i] = x_norm[:,y_t_s_ind[0,i]]
hkl.dump([x, y_t, x_norm, x_n_s, y_t_s], "norm_red_wine_2.hkl")