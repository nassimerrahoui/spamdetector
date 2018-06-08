# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from spamdetector.cleanData.Statistics import Statistics
import numpy as np


class Knn:

    def __init__(self, df, names, test_size, neighbors):
        self.emailsClean = pd.DataFrame(df, columns=names)
        self.test_size = test_size
        self.neighbors = neighbors
        self.statistics = Statistics()


    def getX(self):
        np.set_printoptions(suppress=True, threshold=np.nan)
        labels = []
        df = (self.statistics.getStatisticsWithWords())\
            .query('var>700 | (average>0.1 & var<=300) | index=="is_spam"')
        for i in df.index:
            labels.append(i)
        X = self.emailsClean
        X = X.drop(labels=labels, axis=1)
        print(type(X))
        return X #<class 'pandas.core.frame.DataFrame'>

    def getY(self):
        y = self.emailsClean["is_spam"]
        y = y.transpose()
        return y #<class 'pandas.core.series.Series'>

    def standardisation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.getX(), self.getY(), test_size=self.test_size)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        classifier = KNeighborsClassifier(n_neighbors=self.neighbors)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print()
        print("************************* Knn Results *****************************")
        print("matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("rapport de classification :")
        print(metrics.classification_report(y_test, y_pred))
        print("score de précision :")
        print(metrics.accuracy_score(y_test, y_pred))
        return metrics.accuracy_score(y_test, y_pred)
