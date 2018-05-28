# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

FILE_spambase_header_names = '../rawData/spambase.header_names'
FILE_spambase_data = '..//rawData//spambase.data'

names = []

with open(FILE_spambase_header_names, 'rt') as f:
    names = [name.strip() for name in f.readlines()]

spambase_data = pd.read_csv(FILE_spambase_data, names=names)
df = pd.DataFrame(spambase_data, columns=names)
#pprint.pprint(df)

def data_spam():
    return df[df['is_spam'] == 1].drop(['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'is_spam'], 1)#.describe().unstack()
def data_no_spam():
    return df[df['is_spam'] == 0].drop(['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'is_spam'], 1)

print(".............................................")
mots = ["char_freq_$", "char_freq_!", "word_freq_order", "word_freq_free", "word_freq_money", "word_freq_receive",
        "word_freq_000", "word_freq_george", "word_freq_650", "word_freq_lab",
        "word_freq_labs", "word_freq_edu", "word_freq_conference", "word_freq_meeting"]

data_X = df[names].loc[0:(int)((len(df)-1)*0.8)].values
data_X_predict= df[names].loc[(int)((len(df))*0.8):len(df)].values
print((int)((len(df)-1)*0.8))
data_Y = df["is_spam"].loc[0:(int)((len(df)-1)*0.8)]
data_Y_predict = df["is_spam"].loc[(int)((len(df))*0.8):len(df)]
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-")



X = data_X
print(X)
print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
Y = data_Y
print(Y)
print("------------------------")
clf = GaussianNB()
clf.fit(X, Y)
print("1 : ", clf.predict(data_X_predict))

print(accuracy_score(data_Y_predict, clf.predict(data_X_predict)))