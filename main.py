import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

filename = "czerwone_sortowane_t.txt"
data = np.loadtxt(filename, delimiter=",", dtype=str)
data = np.delete(data, 0, axis=0)
licznik=0

for i in range(0,len(data)):
    if i==len(data)-licznik:
        break
    for j in range(0,len(data[i])):
        if data[i][j] == "":
            licznik+=1
            data=np.delete(data,i,axis=0)
            if data[i][j] == "":
                data = np.delete(data, i, axis=0)

x = data[:, 0:-1].astype(float).T
y_t = data[:, -1].astype(float)
y_t = y_t.reshape(1,y_t.shape[0])

x_min = x.min(axis=1)
x_max = x.max(axis=1)

x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)

for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
    (x[i,:]-x_min[i]) + x_norm_min

