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
            self.names = [name.strip() for name in f.readlines()]
        self.spambase_data = pd.read_csv(FILE_spambase_data, names=self.names)
        self.df = pd.DataFrame(self.spambase_data, columns=self.names)
        self.selected_words = ["char_freq_$", "char_freq_!", "word_freq_order", "word_freq_free", "word_freq_money", "word_freq_receive",
        "word_freq_000", "word_freq_george", "word_freq_650", "word_freq_lab",
        "word_freq_labs", "word_freq_edu", "word_freq_conference", "word_freq_meeting"]


    def getNames(self):
        return self.names

    def getSpambase_data(self):
        return self.spambase_data

    def getDf(self):
        return self.df

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
