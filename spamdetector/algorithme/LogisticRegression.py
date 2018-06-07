from spamdetector.cleanData import Training
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

class LogisticRegressionAlgo:

    def __init__(self, tr):
        self.lR = LogisticRegression()
        self.tr = tr
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.tr.divide()

    def result(self):
        self.lR.fit(self.X_train, self.Y_train)
        predict_lR = self.lR.predict(self.X_test)

        return metrics.accuracy_score(self.Y_test, predict_lR)

