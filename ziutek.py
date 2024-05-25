import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer


class wsteczna:
    ##Przypisanie parametrow w sieci
    def __init__(self, x, y_t, K1, K2, lr, err_goal, \
                 disp_freq, max_epoch):
        self.x = x
        self.L = self.x.shape[0]
        self.y_t = y_t
        self.K1 = K1
        self.K2 = K2
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.max_epoch = max_epoch
        self.K3 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        ##ustalenie wag oraz biasow losowane za pomoca biblioteki net
        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        ##usunieci starych wag
        hkl.dump([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3],
                 'wagi3w.hkl')
        ##wczytanie wag
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = hkl.load('wagi3w.hkl')
        self.SSE = 0
        self.lr_vec = list()


    def predict(self, x):
        ##mnozenie danych wejsciowych przez wagi
        n = np.dot(self.w1, x)
        ##obliczanie wyjsc w warstwach
        self.y1 = net.tansig(n, self.b1 * np.ones(n.shape))
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig(n, self.b2 * np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.purelin(n, self.b3 * np.ones(n.shape))
        return self.y3


    def train(self, x_train, y_train):
        ##przejscie w zasiegu epoch
        for epoch in range(1, self.max_epoch + 1):
            ##dane wyjsciowe
            self.y3 = self.predict(x_train)
            ##obliczanie bledu
            self.e = y_train - self.y3
            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100:
                break

            if np.isnan(self.SSE):
                break
                ##obliczanie wag i biasów
            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(self.x, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1 += self.dw1
            self.b1 += self.db1
            self.w2 += self.dw2
            self.b2 += self.db2
            self.w3 += self.dw3
            self.b3 += self.db3

            self.SSE_vec.append(self.SSE)


    def train_CV(self, CV, skfold):
        PK_vec = np.zeros(CVN)
        ##trenowanie sieci i sprawdzanie poprawnosci wynikow
        for i, (train, test) in enumerate(skfold.split(self.data,np.squeeze(self.target)), start=0):
            x_train, x_test = self.data[train], self.data[test]
            y_train, y_test = np.squeeze(self.target)[train],np.squeeze(self.target)[test]

            self.train(x_train.T, y_train.T)
            result = mlpnet.predict(x_test.T)

            n_test_samples = test.size
            PK_vec[i] = sum((abs(result - y_test) < 0.5).astype(int)[0] /
                            n_test_samples) * 100

            PK = np.mean(PK_vec)
            return PK

# zaladowanie przygotowanych danych
x,y_t,x_norm,x_n_s,y_t_s  = hkl.load("norm_white_wine.hkl")

max_epoch = 1000
err_goal = 0.25
disp_freq = 100
lr = 0.001
K1 = 5
K2 = 15

start = timer()

data = x.T
target = y_t

CVN = 4
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)

##trenowanie sieci i sprawdzanie poprawnosci wynikow
for i, (train, test) in enumerate(skfold.split(data, np.squeeze(target)),
                                  start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]

    mlpnet = wsteczna(x_train.T, y_train, K1, K2, lr, err_goal,disp_freq, max_epoch)
    mlpnet.train(x_train.T, y_train.T)
    result = mlpnet.predict(x_test.T)

    n_test_samples = test.size
    PK_vec[i] = (1 - sum((abs(result - y_test) >= 0.5).astype(int)[0]) / n_test_samples) * 100


    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i],n_test_samples))

print("Execution time:", timer() - start)
PK = np.mean(PK_vec)
print("PK {}".format(PK))