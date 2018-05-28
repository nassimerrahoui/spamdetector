# -*- coding: utf-8 -*-
import csv
import pprint
import pandas as pd

FILE_spambase_header_names = '../rawData/spambase.header_names'
FILE_spambase_data = '..//rawData//spambase.data'

names = []

with open(FILE_spambase_header_names, 'rt') as f:
    names = [name.strip() for name in f.readlines()]

spambase_data = pd.read_csv(FILE_spambase_data, names=names)
df = pd.DataFrame(spambase_data, columns=names)

def data_spam():
    return df[df['is_spam'] == 1].describe().unstack()

def data_no_spam():
    return df[df['is_spam'] == 0].describe().unstack()

pprint.pprint(data_spam().max)
