# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


names = []


class Training:

    def __init__(self,FILE_spambase_header_names, FILE_spambase_data):
        with open(FILE_spambase_header_names, 'rt') as f:
            self.names = [name.strip() for name in f.readlines()]
        self.spambase_data = pd.read_csv(FILE_spambase_data, names=self.names)

        self.target = self.spambase_data[self.spambase_data.columns[-1]]  # la columns spam
        self.x = self.spambase_data.drop(self.spambase_data.columns[-1], axis=1)  # sans la columns spam

    def getTarget(self):
        return self.target

    def getX(self):
        return self.x

    def divide(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.x,
                                                            self.target,
                                                            test_size=0.2,#0.2
                                                            random_state=50)#42
        return X_train, X_test, Y_train, Y_test
