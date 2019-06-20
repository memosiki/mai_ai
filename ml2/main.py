import time
import numpy as np
import sklearn
from sklearn import neighbors
from algorithm import *
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def compare_methods(train_data, train_label, test_data, test_label):
    n_neighbors = 21

    # sklearn
    start_time = time.time()
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='brute')
    clf.fit(train_data, train_label)
    print("sklearn")
    print("precision: ", clf.score(test_data, test_label))
    print("время работы", (time.time() - start_time))

    # Моя реализация
    start_time = time.time()
    model = KNN(n_neighbors)
    model.fit(train_data, train_label)
    pred = model.predict(test_data)
    pred = np.array(pred)
    print("Собственная реализация")
    print("precision: ", (pred == test_label).mean())
    print("время работы", (time.time() - start_time))


def stocks():
    # Датасет с акциями
    print('Акции')
    with open('IXIC-redacted.csv', 'r') as f:
        reader = csv.reader(f)
        df = list(reader)

    # Данные
    X = []
    y = []
    # пропускаем первую строку с подписями столбцов
    for elem in df[1:]:
        # пропускаем первый столбец с индексами строки
        X.append(elem[1:len(elem) - 1])
        y.append(elem[len(elem) - 1])
    # приводим типы из str в соответсвующие
    y = np.array([int(elem) for elem in y])
    X = np.array([[float(elem) for elem in row] for row in X])

    # разделяем данные на обучение и проверку
    (train_data, train_label), (test_data, test_label) = \
        split_data(X, y, 0.8)
    # train_data = test_data = X
    # train_label = test_label = y

    compare_methods(train_data, train_label, test_data, test_label)


def test_text():
    # Текстовые данные
    print('Текстовые данные')
    df = pd.read_csv("text.csv")
    classes = ['python', 'c++']
    df = df[df.tags.isin(classes)]
    # Приводим текст к разреженной матрице с помощью sklearn
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['post'])
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X_train_counts).todense()
    X = [np.asarray(row)[0] for row in X]
    y = [classes.index(elem) for elem in df['tags']]
    (_, _), (test_data, test_label) = split_data(X, y, 0.1)
    (_, _), (train_data, train_label) = split_data(X, y, 0.1)
    compare_methods(train_data, train_label, test_data, test_label)


def regression_check():
    print("Полиномиальная регрессия")
    start_time = time.time()
    df = pd.read_csv('IXIC-redacted.csv')
    X = np.array(df[['Date', 'Open', 'High', 'Low', 'Volume']].to_numpy())
    y = df['Close'].to_numpy()
    (train_data, train_value), (test_data, test_value) = \
        split_data(X, y, 0.8)
    model = make_pipeline(PolynomialFeatures(1), Ridge())
    model.fit(train_data, train_value)
    res = model.predict(test_data)
    e = sklearn.metrics.mean_squared_error(res, test_value)
    print("sklearn")
    print('Среднеквадратичная ошибка', e)
    print("время работы", (time.time() - start_time))
    print()
    regr = PolynomialRegression(1)
    regr.fit(train_data, train_value)
    res = regr.predict(test_data)
    e = sklearn.metrics.mean_squared_error(res, test_value)
    print("Собственная реализация")
    print('Среднеквадратичная ошибка', e)
    print("время работы", (time.time() - start_time))


if __name__ == '__main__':
    # stocks()
    # test_text()
    regression_check()
