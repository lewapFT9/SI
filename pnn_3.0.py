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

    def fit(self, X, y, activ):
        self.classes = np.unique(y)
        self.models = {}
        for cls in self.classes:
            kde = KernelDensity(kernel=activ, bandwidth=self.bandwidth)
            kde.fit(X[y == cls])
            self.models[cls] = kde
            #print(self.models[cls].get_params())

    def predict(self, X):
        log_probs = np.array([model.score_samples(X) for model in self.models.values()]).T
        return self.classes[np.argmax(log_probs, axis=1)]


data = pd.read_csv('white_quality.csv', sep=";" )
# print(data.columns)

# Podział danych na klasy
# class_1 = data[data['quality'] == 3]
# class_2 = data[data['quality'] == 4]
# class_3 = data[data['quality'] == 5]
# class_4 = data[data['quality'] == 6]
# class_5 = data[data['quality'] == 7]
# class_6 = data[data['quality'] == 8]
# class_7 = data[data['quality'] == 9]

# class_1.to_csv('class_1.csv', index=False)
# class_2.to_csv('class_2.csv', index=False)
# class_3.to_csv('class_3.csv', index=False)
# class_4.to_csv('class_4.csv', index=False)
# class_5.to_csv('class_5.csv', index=False)
# class_6.to_csv('class_6.csv', index=False)
# class_7.to_csv('class_7.csv', index=False)

# Wczytanie podzielonych danych z plików CSV
# class_1 = pd.read_csv('class_1.csv')
# class_2 = pd.read_csv('class_2.csv')
# class_3 = pd.read_csv('class_3.csv')
# class_4 = pd.read_csv('class_4.csv')
# class_5 = pd.read_csv('class_5.csv')
# class_6 = pd.read_csv('class_6.csv')
# class_7 = pd.read_csv('class_7.csv')

# Połączenie danych w jeden zbiór treningowy
#data = pd.concat([class_1, class_2, class_3, class_4, class_5, class_6, class_7], axis=0)

X = data.drop('quality', axis=1)
y = data['quality']
join_scores = []
splits = [0.1, 0.2, 0.3, 0.4, 0.5]
for k in range(len(splits)):


    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splits[k], random_state=46)

    scaler = StandardScaler()

    #print(X_test)
    # Normalizacja danych
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #y_test = scaler.transform(y_test)
    # Trenowanie modelu PNN
    bandwidth = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    activ_f = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    scores = [[], [], [], [], [], []]


    #print(X_test)
    for i in range(len(activ_f)):
        for j in range(len(bandwidth)):

            pnn = PNN(bandwidth= bandwidth[j])
            pnn.fit(X_train, y_train,activ_f[i])

            # Predykcje
            y_pred = pnn.predict(X_test)

            # for i in range(len(y_test)):
            #     print("Próbka {}: Predykcja - {}, Oczekiwana - {}".format(i+1, y_pred[i], y_test.iloc[i]))

            # Ocena modelu
            accuracy = accuracy_score(y_test, y_pred)
            # print(f'Accuracy: {accuracy * 100:.2f}%')
            scores[i].append(accuracy)
        join_scores.append(scores)

    # print(scores)

    plt.plot(bandwidth, scores[0], 'b', bandwidth, scores[1], "r",
             bandwidth, scores[2], "g", bandwidth, scores[3], "y", bandwidth, scores[4], "m",
             bandwidth, scores[5], "c")
    plt.xlabel("bandwidth")
    plt.ylabel("accuracy")
    ax = plt.gca()  # Pobieranie aktualnej osi
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.legend(activ_f, loc="lower right",
               bbox_to_anchor=(1.4, 0), borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.title(f"data split {splits[k]}")
    plt.grid(True)
    plt.show()
    #print(k+1)


