from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from spamdetector.algorithme.Backpropagation import Backpropagation
from spamdetector.algorithme.Iris_df import Iris_df
from spamdetector.algorithme.KernelSVM import KernelSVM
from spamdetector.algorithme.KNene import metrics
from spamdetector.algorithme.LogisticRegression import LogisticRegression
from spamdetector.algorithme.NaiveBayes import NaiveBayes
from spamdetector.algorithme.RandomForest import RandomForest

from spamdetector.unsupervised.DBSCAN import DBSCAN
from spamdetector.unsupervised.Tsne import Tsne
from spamdetector.unsupervised.Unsupervised import Unsupervised


FILE_spambase_header_names = r'./rawData/spambase.header_names'
FILE_spambase_data = r'./rawData/spambase.data'
FILE_img = r'./outPut/graph'


class Algos:
    def __init__(self, cd, ld, tr):
        self.cleanData = cd
        self.load_data = ld
        self.traning = tr

        self.classifiers = [
            ("Back propagation", Backpropagation(self.training,FILE_spambase_data)),
            ("Back propagation", Backpropagation(self.training, FILE_spambase_data)),
        ]


    def run(self):
        return self.run_algos(self.classifiers)


    def run_algos(self, algo_list):
        confusion_matrixes = {}
        X_train, X_test, Y_train, Y_test = self.tr.divide()

        for (name, algo) in algo_list:

            algo.fit(X_train, Y_train)
            predict = algo.predict(X_test)

            accuracy = round(accuracy_score(Y_test, predict ), 4)*100
            precision = round(precision_score(Y_test, predict ), 4)*100
            recall = round(recall_score(Y_test, predict ), 4)*100
            f1_score_ = round(f1_score(Y_test, predict ), 4)*100
            auc = round(roc_auc_score(Y_test, predict ), 4)*100
            # probas = algo.predict_proba(main.X_test)

            print("algo : ", name, ':  Accuracy - ', accuracy, ', Precision - ', precision, ', Recall - ', recall,
                  ', F1_score - ', f1_score_, ' AUC - ', auc)

            confusion_matrixes[name] = (confusion_matrix(Y_test, predict))

        return confusion_matrixes
