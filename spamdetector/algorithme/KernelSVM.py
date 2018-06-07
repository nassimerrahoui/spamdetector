import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from spamdetector.cleanData.Statistics import Statistics
from sklearn.metrics import classification_report, confusion_matrix


class KernelSVM:

    def __init__(self, df, names, test_size):
        self.emailsClean = pd.DataFrame(df, columns=names)
        self.test_size = test_size
        self.statistics = Statistics()

    def getX(self):
        labels = []
        df = (self.statistics.getStatisticsWithWords()) \
            .query('var>700 | (average>0.1 & var<=300) | index=="is_spam"')
        for i in df.index:
            labels.append(i)
        X = self.emailsClean
        X = X.drop(labels=labels, axis=1)
        return X

    def getY(self):
        y = self.emailsClean["is_spam"]
        y = y.transpose()
        return y

    def results(self, kernel):
        X_train, X_test, y_train, y_test = train_test_split(self.getX(), self.getY(), test_size=self.test_size)

        svclassifier = SVC(kernel=kernel)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        print()
        print("************************* Kernel SVM Results *****************************")
        print("rapport de classification :")
        print(metrics.classification_report(y_test, y_pred))
        print("score de précision :")
        print(metrics.accuracy_score(y_test, y_pred))
