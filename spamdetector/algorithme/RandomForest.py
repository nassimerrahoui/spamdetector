from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score,roc_auc_score
import time
tmps1=time.time()

class RandomForest:
    def __init__(self, tr):
        self.lR = RandomForestClassifier(n_jobs = -1, random_state=9,oob_score = False, bootstrap=True)
        self.tr = tr


    def result(self):
        X_train, X_test, Y_train, Y_test = self.tr.divide()
        self.lR.fit(X_train, Y_train)
        predict_lR = self.lR.predict(X_test)
        accuracy_RF = accuracy_score(Y_test, predict_lR)
        precision_RF = precision_score(Y_test, predict_lR)
        recall_RF = recall_score(Y_test, predict_lR)
        f1_score_RF = f1_score(Y_test, predict_lR)
        auc_RF = roc_auc_score(Y_test, predict_lR)
        print('accuracy ', accuracy_RF, 'precision ', precision_RF, 'recall ', recall_RF, 'f1_score ', f1_score_RF,
              'auc ', auc_RF)

tmps2=time.time()-tmps1
print("Temps d'execution = ", tmps2)
