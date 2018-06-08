# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


FILE_spambase_header_names = 'rawData/spambase.header_names'
FILE_spambase_data = 'rawData/spambase.data'

names = []
class CleanData:
    def __init__(self):
        with open(FILE_spambase_header_names, 'rt') as f:
            self.names = [name.strip() for name in f.readlines()] #list
        self.spambase_data = pd.read_csv(FILE_spambase_data, names=self.names) #dataframe
        self.df = pd.DataFrame(self.spambase_data, columns=self.names) #pareil que précédent
        self.numpy_array = self.spambase_data.values #numpy.ndarray
        self.selected_words = ["char_freq_$", "char_freq_!", "word_freq_order", "word_freq_free", "word_freq_money", "word_freq_receive",
        "word_freq_000", "word_freq_george", "word_freq_650", "word_freq_lab",
        "word_freq_labs", "word_freq_edu", "word_freq_conference", "word_freq_meeting"]

    def getClean_df(self):
        return self.spambase_data.pop('is_spam')

    def data(self):
        return self.spambase_data.drop('is_spam',1)

    def names_(self):
        names = self.getNames()
        del names[-1]
        del names[54]
        del names[26]
        return names

    def target(self):
        return self.getSpambase_data()['is_spam']

    def getNames(self):
        return self.names #<class 'list'> only labels

    def getSpambase_data(self):
        return self.spambase_data #<class 'pandas.core.frame.DataFrame'> labels+row numbers+data

    def getDf(self):
        return self.df #<class 'pandas.core.frame.DataFrame'> labels+row numbers+data

    def getNumpy_array(self):
        return self.numpy_array #<class 'numpy.ndarray'> only data (whole data)

    def getSelected_words(self):
        return self.selected_words

    def setSelected_words(self, selected_words):
        self.selected_words = selected_words

    def data_spam(self, df):
        return df[df['is_spam'] == 1].drop(
            ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'is_spam'], 1)

    def data_no_spam(self, df):
        return df[df['is_spam'] == 0].drop(
            ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'is_spam'], 1)
