from pydoc import classname
import numpy as np
import pandas as pd
import statistics
import sklearn.model_selection
import skfuzzy as fuzz
import time

# function to find accuracy between predicted value and testing data
def accuracy(y_pre, y_true):
    n = len(y_true)
    count = 0
    for i in range(n):
        # print(type(y_true), type(y_pre))
        if np.equal(y_true[i], y_pre).any() == True:
            count = count + 1
    acc = count / n
    return acc


def one_step_knn(traindata, ytrain, testdata, classnum, alpha, beta):
    X = traindata
    Y = testdata
    n, d = X.shape[0], X.shape[1]
    m, d1 = Y.shape[0], Y.shape[1]
    eps = np.finfo(float).eps
    # tic=time.time()

    classnumber = classnum

    # group by fuzzy c means clustering
    index = np.zeros((classnumber, n))
    IGg = np.zeros((classnumber, n))
    fcm_result = fuzz.cluster.cmeans(X.T, classnumber, 2.0, 0.001, 25)
    U = fcm_result[1]

    for i in range(classnumber):
        idx = U[i, :] > 0.01
        idxx = np.where(idx)[0]
        index[i, 0 : len(idxx)] = idxx
        IGg[i] = idx

    # initilisation
    result = []
    W = np.random.rand(n, m)
    iter = 0
    obji = 1
    Wsum = np.sum(W, axis=1)
    obj = []
    while 1:
        for i in range(1, n):
            FF = np.zeros((n))
            Groupsum = 0
            for ii in range(0, classnumber):
                indexx = []
                indexx = np.nonzero(U[ii] > 0.01)
                group = np.array(Wsum[indexx])
                Groupsum = Groupsum + (
                    IGg[ii, i] * np.linalg.norm(group)
                ) / np.linalg.norm(W[i, :])

            FF[i] = Groupsum
        F = np.diag(FF)

        # Çó³öW
        dn = np.zeros((n))

        for i in range(1, n):
            dn[i] = np.sqrt(sum((np.sum(np.multiply(W, W), 2 - 1) + eps))) / np.sum(
                W[i, :]
            )

        N = np.diag(dn)
        W = np.linalg.solve((np.dot(X, X.T) + alpha * N + beta * F), (np.dot(X, Y.T)))
        W[W < np.mean(W)] = 0
        Wi = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        W21 = np.sum(Wi)
        Wd = np.sum(W, 1)
        obj.append(
            np.linalg.norm(Y.T - np.dot(X.T, W), "fro") ** 2
            + alpha * W21
            + beta * np.dot(Wd.T, np.dot(F, Wd))
        )

        cver = abs((obj[iter] - obji) / obji)
        obji = obj[iter]
        iter += 1
        if (cver < eps and iter > 2) or iter == 2:
            print("finished")
            break

        tic = time.time()
        weight = np.zeros(classnumber)
        Labels = np.zeros(m, dtype=int)
        for i in range(m):
            idx = np.where(W[:, i] != 0)[0]
            for j in range(classnum):
                if idx.size > 0:
                    idxnum = np.where(y_train[idx] == j)
                    weight[j] = np.sum(W[idxnum, i])
            if np.all(np.isnan(weight)):
                unique, counts = np.unique(y_train, return_counts=True)
                argmax = np.argmax(counts)
                Labels[i] = unique[argmax]
            if np.all(~np.isnan(weight)):
                idxmax = np.where(weight == np.max(weight))[0]
                if len(idxmax) > 1:
                    Labels[i] = idxmax[0]
                else:
                    Labels[i] = idxmax
        elapsed = time.time() - tic
        return Labels, elapsed


dataset = pd.read_csv("glass.csv")
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
kf = sklearn.model_selection.KFold(n_splits=10, shuffle=False)

epochs = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    label, tim = one_step_knn(X_train, y_train, X_test, 6, 10000, 1)
    # print(y_train)
    print(f"epochs: {epochs}, time: {round(tim, 6)}s {label}")
    epochs += 1
print("Accuracy:", accuracy(label, y_train))
