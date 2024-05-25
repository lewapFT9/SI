import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hickle as hkl

class PNN:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y, activ, met):
        self.classes = np.unique(y)
        self.models = {}
        for cls in self.classes:
            kde = KernelDensity(kernel=activ, bandwidth=self.bandwidth, metric= met)
            kde.fit(X[y == cls])
            self.models[cls] = kde


    def predict(self, X):
        log_probs = np.array([model.score_samples(X) for model in self.models.values()]).T
        return self.classes[np.argmax(log_probs, axis=1)]


data = pd.read_csv('wine_red.csv', sep=";")


X = data.drop('quality', axis=1)
y = data['quality']
splits = [0.1, 0.2, 0.3, 0.4, 0.5]


for k in range(len(splits)):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splits[k], random_state=46)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    bandwidth = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    activ_f = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    scores_1 = [[], [], [], [], [], []]
    max_score_1 = 0
    max_score_2 = 0
    opt_f = ""
    opt_met = ""

    for i in range(len(activ_f)):
        for j in range(len(bandwidth)):

            pnn = PNN(bandwidth= bandwidth[j])
            pnn.fit(X_train, y_train,activ_f[i],"euclidean")

            y_pred = pnn.predict(X_test)

            # for i in range(len(y_test)):
            #     print("Próbka {}: Predykcja - {}, Oczekiwana - {}".format(i+1, y_pred[i], y_test.iloc[i]))

            accuracy = accuracy_score(y_test, y_pred)
            scores_1[i].append(accuracy)
            if accuracy > max_score_1:
                max_score_1 = accuracy
                opt_f = activ_f[i]




    plt.plot(bandwidth, scores_1[0], 'b', bandwidth, scores_1[1], "r",
             bandwidth, scores_1[2], "g", bandwidth, scores_1[3], "y", bandwidth, scores_1[4], "m",
             bandwidth, scores_1[5], "c")
    plt.xlabel("bandwidth")
    plt.ylabel("accuracy")
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.legend(activ_f, loc="lower right",
               bbox_to_anchor=(1.4, 0), borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.title(f"data_split: {splits[k]}"
              f"max_score: {max_score_1}"
              f"opt_f: {opt_f}")
    plt.grid(True)
    plt.show()

    metric = ['euclidean', 'manhattan', 'chebyshev']
    scores_2 = [[], [], []]
    for m in range(len(metric)):
        for b in range(len(bandwidth)):
            pnn_2 = PNN(bandwidth=bandwidth[b])
            pnn_2.fit(X_train, y_train, "gaussian", metric[m])
            y_pred = pnn_2.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores_2[m].append(accuracy)
            if accuracy > max_score_2:
                max_score_2 = accuracy
                opt_met = metric[m]

    plt.plot(bandwidth, scores_2[0], 'b', bandwidth, scores_2[1], "r",
             bandwidth, scores_2[2], "g")
    plt.xlabel("bandwidth")
    plt.ylabel("accuracy")
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.legend(metric, loc="lower right",
               bbox_to_anchor=(1.4, 0), borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.title(f"data split {splits[k]}"
              f"max_score: {max_score_2}"
              f"opt_met: {opt_met}")
    plt.grid(True)
    plt.show()

    scores_3 = []

    for bnd in range(len(bandwidth)):
        pnn = PNN(bandwidth=bandwidth[bnd])
        pnn.fit(X_train, y_train, "gaussian", "euclidean")
        y_pred = pnn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores_3.append(accuracy)


    plt.plot(bandwidth, scores_3)
    plt.xlabel("bandwidth")
    plt.ylabel("accuracy")
    plt.title(f"data split {splits[k]}")
    plt.grid(True)
    plt.show()









