from spamdetector.cleanData.CleanData import CleanData
from spamdetector.algorithme.NaiveBayes import NaiveBayes
from spamdetector.algorithme.KNene import Knn
from spamdetector.algorithme.KernelSVM import KernelSVM

if __name__ == '__main__':
    clean_data = CleanData()

    #
    naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNumpy_array(), 0.20) #20% de predict
    naiveBayes.result()

    # Neighboors define classes of a row
    knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    knn.standardisation()

    # Kernel SVM seperate classes by a frontier in a dimension with Kernel method
    # linear trace straight frontier
    # rbf trace straight frontier but accept errors
    kernelSVM = KernelSVM(clean_data.getDf(), clean_data.getNames(), 0.20)
    kernelSVM.results("linear")  # you can set at "linear" or "rbf"
