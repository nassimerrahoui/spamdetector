# -*- coding: utf-8 -*-

from pathlib import Path
#from sklearn import datasets
import pandas as pd

#iris = datasets.load_iris()
data = str(Path.home()) + '\IdeaProjects\dataScience\dataset\spambase.data'

#file = open(data, 'r', delimiter=',')

dataPanda = pd.read_csv(data)

nb = 0

print(dataPanda.iloc[:, -1])

print("nb : ",  nb)