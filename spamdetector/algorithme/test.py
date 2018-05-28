# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = np.array([[-1, -1], [-2, -1]])
Y = np.array([0,1])
clf = GaussianNB()
clf.fit(X, Y)
print("1 : ", clf.predict([[-0.8, -1]]))
#clf_pf = GaussianNB()
#clf_pf.partial_fit(X, Y, np.unique(Y))
#print("1 : ", clf_pf.predict([[-0.8, -1]]))

