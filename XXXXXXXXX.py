import numpy as np
import pandas as pd


data = np.loadtxt("SSE_VEC.txt")
print(data)

data = data.T

np.savetxt("SSE_VEC.txt", data, delimiter= ",")

print(data)

tab = []

for i in range (len(data)):
    tab.append(i+1)

print(tab)

np.savetxt("iksy.txt",tab, delimiter= ",")

