from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import math
import random


def pobieranieDanych():
    bazaDanych = []
    with open('bazaDanych.txt', 'r') as file:
        for line in file:
            bazaDanych.append(list(map(int, line.strip())))

    etykietyKlasy = []
    for i in range(10):
        for j in range(10):
            etykietyKlasy.append(j)

    los = random.randint(0, 79)
    xTest = [bazaDanych.pop(los) for i in range(20)]
    yTest = [etykietyKlasy.pop(los)for i in range(20)]
    xTrain = bazaDanych
    yTrain = etykietyKlasy
    return np.array(xTest), np.array(xTrain), np.array(yTest), np.array(yTrain)


def treningLVQ(X, y, a, b, max_ep, min_a, e):
    c, train_idx = np.unique(y, True)
    r = c
    W = X[train_idx].astype(np.float64)
    idx = np.array([i for i in np.arange(X.shape[0]) if i not in train_idx])
    X = X[idx, :]
    y = y[idx]
    ep = 0
    while ep < max_ep and a > min_a:
        for i, x in enumerate(X):
            d = [math.sqrt(sum((w - x) ** 2)) for w in W]
            min_1 = np.argmin(d)
            min_2 = 0
            dc = float(np.amin(d))
            dr = 0
            min_2 = d.index(sorted(d)[1])
            dr = float(d[min_2])
            if c[min_1] == y[i] and c[min_1] != r[min_2]:
                W[min_1] = W[min_1] + a * (x - W[min_1])

            elif c[min_1] != r[min_2] and y[i] == r[min_2]:
                if dc != 0 and dr != 0:

                    if min((dc / dr), (dr / dc)) > (1 - e) / (1 + e):
                        W[min_1] = W[min_1] - a * (x - W[min_1])
                        W[min_2] = W[min_2] + a * (x - W[min_2])
            elif c[min_1] == r[min_2] and y[i] == r[min_2]:
                W[min_1] = W[min_1] + e * a * (x - W[min_1])
                W[min_2] = W[min_2] + e * a * (x - W[min_2])
        a = a * b
        ep += 1
    return W, c


def testLVQ(x, W):
    W, c = W
    d = [math.sqrt(sum((w - x) ** 2)) for w in W]
    return c[np.argmin(d)]


def wyswietlanieStatystyk(labels, preds):
    print("Wskaźnik Precyzji: {}".format(precision_score(labels, preds, average='weighted')))
    print("Wskaźnik Odtworzenia: {}".format(recall_score(labels, preds, average='weighted')))
    print("Dokładność: {}".format(accuracy_score(labels, preds)))
    print("Wskaźnik F1: {}".format(f1_score(labels, preds, average='weighted')))


x_test, x_train, y_test, y_train = pobieranieDanych()

W = treningLVQ(x_train, y_train, 0.2, 0.5, 100, 0.001, 0.2)
#print(W)

predicted = []
for i, true_label in zip(x_test, y_test):
    prediction = testLVQ(i, W)
    predicted.append(prediction)
    correctness = "+" if prediction == true_label else "-"
    print(f'Ciąg binarny: {i}. Przewidywania: {prediction}. Rzeczywistość: {true_label}.')

wyswietlanieStatystyk(y_test, predicted)
