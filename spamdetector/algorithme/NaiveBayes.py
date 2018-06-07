# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

class NaiveBayes:
    def __init__(self, df, names, test_size):
        np.set_printoptions(suppress=True, threshold=np.nan)

        #Select data labels instead of mails
        names2 = names.transpose()
        spam_label_index = len(names2)-1
        names3 = np.delete(names2, spam_label_index, 0)
        names3 = names3.transpose() #Remettre dans le bon sens
        print(names3[0]) #1st mail without spam

        self.test_size = test_size
        self.data_X = names3
        self.data_Y = names2[57]

    def result(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data_X, self.data_Y, test_size=self.test_size)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print()
        print("************************* Naive Bayes Results *****************************")
        print("matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("rapport de classification :")
        print(metrics.classification_report(y_test, y_pred))
        print("score de précision :")
        print(metrics.accuracy_score(y_test, y_pred))

    def getData_X(self):
        return self.data_X

    def setData_X(self, data_X):
        self.data_X = data_X

    def getData_Y(self):
        return self.data_Y

    def setData_Y(self, data_Y):
        self.data_Y = data_Y



