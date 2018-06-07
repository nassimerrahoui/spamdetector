from spamdetector.cleanData.CleanData import CleanData
from spamdetector.cleanData.Training import Training
from spamdetector.algorithme.LogisticRegression import LogisticRegressionAlgo
from spamdetector.algorithme.RandomForest import RandomForest
from spamdetector.algorithme.Backpropagation import Backpropagation
from spamdetector.algorithme.NaiveBayes import NaiveBayes
from spamdetector.algorithme.KNene import Knn

FILE_spambase_header_names = r'./rawData/spambase.header_names'
FILE_spambase_data = r'./rawData/spambase.data'


if __name__ == '__main__':
    clean_data = CleanData()
    training = Training(FILE_spambase_header_names, FILE_spambase_data)

    """Backpropagation"""
    backpropagation = Backpropagation(training,FILE_spambase_data)
    backpropagation.data_file()


    """LogisticRegression"""
    lRalgo = LogisticRegressionAlgo(training)
    #print(lRalgo.accuracy())

    """RandomForest"""
    randomForest = RandomForest(training)
    #print(randomForest.result())

    """NaiveBayes"""
    #naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNames())
    #print(naiveBayes.detailed_result())

    """Knn"""
    #knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    #knn.standardisation()
