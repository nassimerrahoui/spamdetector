from spamdetector.cleanData.CleanData import CleanData
from spamdetector.cleanData.Training import Training
from spamdetector.unsupervised.Classification_hiérarchique import Classification_hiérarchique
from spamdetector.unsupervised.Tsne import Tsne
from spamdetector.unsupervised.DBSCAN import DBSCANalgo
from spamdetector.algorithme.RandomForest import RandomForest
from spamdetector.algorithme.LogisticRegression import LogisticRegressionAlgo

FILE_spambase_header_names = r'./rawData/spambase.header_names'
FILE_spambase_data = r'./rawData/spambase.data'
FILE_img = r'./outPut/graph'


if __name__ == '__main__':
    clean_data = CleanData()
    training = Training(FILE_spambase_header_names, FILE_spambase_data)


    """Backpropagation"""
    #backpropagation = Backpropagation(training,FILE_spambase_data)
    #backpropagation.data_file()

    """LogisticRegression"""
    print("############LogisticRegression############")
    #lRalgo = LogisticRegressionAlgo(training)
    #print(lRalgo.result())
    print("-------------------------------------------")
    print("")

    """RandomForest"""
    print("############RandomForest############")
    #randomForest = RandomForest(training)
    #print(randomForest.result())
    print("-------------------------------------------")
    print("")

    """NaiveBayes"""
    print("############NaiveBayes############")
    #naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNumpy_array(), 0.20)  # 20% de predict
    #naiveBayes.result()
    print("-------------------------------------------")
    print("")

    """Knn"""
    print("############Knn############")
    #knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    #knn.standardisation()
    print("-------------------------------------------")
    print("")

    """Neighboors"""
    print("############Neighboors############")
    # Neighboors define classes of a row
    #knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    #knn.standardisation()
    print("-------------------------------------------")
    print("")

    """Kernel SVM"""
    print("############Kernel SVM############")
    # Kernel SVM seperate classes by a frontier in a dimension with Kernel method
    # linear trace straight frontier
    # rbf trace straight frontier but accept errors
    #kernelSVM = KernelSVM(clean_data.getDf(), clean_data.getNames(), 0.20)
    #kernelSVM.results("linear")  # you can set at "linear" or "rbf"
    print("-------------------------------------------")
    print("")

    """iris_df"""
    print("############iris_df############")
    # Neighboors define classes of a row
    #iris_df = Iris_df(training)
    #iris_df.result()
    print("-------------------------------------------")
    print("")


    """iris_df"""
    print("############iris_df############")
    # Neighboors define classes of a row
    #unsupervised = Unsupervised(training, clean_data)
    #unsupervised.result()
    print("-------------------------------------------")
    print("")

    """Tsne"""
    print("############iris_df############")
    # Neighboors define classes of a row
    #tsne = Tsne(clean_data, FILE_img)
    #tsne.result()
    print("-------------------------------------------")
    print("")

    """DBSCANalgo"""
    print("############iris_df############")
    # Neighboors define classes of a row
    #db = DBSCANalgo(clean_data, FILE_img)
    #db.result()
    print("-------------------------------------------")
    print("")