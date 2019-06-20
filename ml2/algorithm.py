import random
import time

import math
import numpy as np
import scipy as sc
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import issparse
from sklearn.preprocessing import PolynomialFeatures


class KNN(object):
    # параметр k - количество ближайших соседей
    def __init__(self, k):
        self.k = k
        self.train_data = []
        self.classes = []

    def fit(self, X, y):
        # X -- данные
        # y -- соответствующие им классы

        # элемент списка trainData имеет структуру (координаты, класс)
        self.train_data += zip(X, y)

        # множество классов нужно для посчёта уникальных классов
        # и обращения к ним по индексам
        for elem in y:
            if elem not in self.classes:
                self.classes.append(elem)

    def predict(self, X, custom_dist=None):
        # сопоставляет данным -- классы
        num_of_classes = len(self.classes)
        train_data = self.train_data
        test_data = X

        # Евклидово расстояние между двумя точками, есть возможность задавать своё
        def dist(a, b):
            assert a.shape == b.shape  # ошибка если размерности не совпадают
            n = a.shape[0]
            return sum(
                (a[i] - b[i]) ** 2 for i in range(n))

        if custom_dist:
            dist = custom_dist

        test_labels = []
        for test_point in test_data:
            # print(len(test_data));start_time = time.time()
            # подсчитываем расстояния между заданной точкой и ВСЕМИ точками из TrainData
            test_dist = [[dist(test_point, train_data[i][0]), train_data[i][1]]
                         for i in range(len(train_data))]

            # подсчитываем количество точек каждого класса для k ближайших соседей
            stat = [0 for i in range(num_of_classes)]
            for d in sorted(test_dist)[0:self.k]:
                stat[d[1]] += 1

            # выбираем класс, который встречается чаще всего у соседей
            # np.max
            selected_class = sorted(zip(stat, range(num_of_classes)), reverse=True)[0][1]
            #                сортировка по количеству точек с конкретным классом по убыванию
            test_labels.append(selected_class)
            # print("Затрачено времени на точку", (time.time() - start_time))
        return test_labels


class PolynomialRegression:
    def __init__(self, order):
        self.order = order
        self.pf = PolynomialFeatures(order)
        self.coeff = None

    def fit(self, X, y):
        x_matrix = self.pf.fit_transform(X, y)
        #  lstsq Return the least-squares solution to a linear matrix equation.
        # возвращет решение наименьних квадратов для СЛАУ
        self.coeff = np.linalg.lstsq(x_matrix, y, rcond=None)[0]

    def predict(self, X):
        n = X.shape[0]
        x_matrix = self.pf.fit_transform(X)
        Y = np.ndarray(n)
        for j in range(n):
            Y[j] = sum(x_matrix[j, i] * self.coeff[i]
                       for i in range(x_matrix.shape[1]))
        return Y


def split_data(data, labels, test_percent):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for row, label in zip(data, labels):
        if random.random() < test_percent:
            test_data.append(row)
            test_label.append(label)
        else:
            train_data.append(row)
            train_label.append(label)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    return (train_data, train_label), (test_data, test_label)
