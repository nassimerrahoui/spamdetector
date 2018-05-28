# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self, df, names):
        self.data_X = df[names].loc[0:(int)((len(df) - 1) * 0.8)].values
        self.data_X_predict = df[names].loc[(int)((len(df)) * 0.8):len(df)].values
        self.data_Y = df["is_spam"].loc[0:(int)((len(df) - 1) * 0.8)]
        self.data_Y_predict = df["is_spam"].loc[(int)((len(df)) * 0.8):len(df)]

    def detailed_result(self):
        clf = GaussianNB()
        clf.fit(self.data_X, self.data_Y)
        print("1 : ", clf.predict(self.data_X_predict))
        print(accuracy_score(self.data_Y_predict, clf.predict(self.data_X_predict)))

    def result(self):
        clf = GaussianNB()
        clf.fit(self.data_X, self.data_Y)
        return accuracy_score(self.data_Y_predict, clf.predict(self.data_X_predict))

    def getData_X(self):
        return self.data_X

    def setData_X(self, data_X):
        self.data_X = data_X

    def getData_Y(self):
        return self.data_Y

    def setData_Y(self, data_Y):
        self.data_Y = data_Y

    def getData_X_predict(self):
        return self.data_X_predict

    def setData_X_predict(self, data_X_predict):
        self.data_X_predict = data_X_predict

    def getData_Y_predict(self):
        return self.data_Y_predict

    def setData_Y_predict(self, data_Y_predict):
        self.data_Y_predict = data_Y_predict


