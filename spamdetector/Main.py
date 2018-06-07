from spamdetector.cleanData.CleanData import CleanData
from spamdetector.algorithme.NaiveBayes import NaiveBayes
from spamdetector.algorithme.KNene import Knn

if __name__ == '__main__':
    clean_data = CleanData()

    naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNumpy_array(), 0.20) #20% de predict
    naiveBayes.result()

    #knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    #knn.standardisation()
