from numpy.core.tests.test_mem_overlap import xrange

from spamdetector.algorithme.Backpropagation import Backpropagation
from spamdetector.algorithme.KNene import Knn
from spamdetector.algorithme.KernelSVM import KernelSVM
from spamdetector.algorithme.LogisticRegression import LogisticRegressionAlgo
from spamdetector.algorithme.NaiveBayes import NaiveBayes
from spamdetector.algorithme.RandomForest import RandomForest
from spamdetector.cleanData.CleanData import CleanData
from spamdetector.cleanData.Training import Training
from spamdetector.unsupervised.Unsupervised import Unsupervised
from spamdetector.unsupervised.Tsne import Tsne
from spamdetector.cleanData.Load_data import Load_data
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

FILE_spambase_header_names = r'./rawData/spambase.header_names'
FILE_spambase_data = r'./rawData/spambase.data'
FILE_img = r'./outPut/graph'


if __name__ == '__main__':
    clean_data = CleanData()
    training = Training(FILE_spambase_header_names, FILE_spambase_data)
    load_data = Load_data(clean_data)

    """Backpropagation"""
    print("############Backpropagation############")
    backpropagation = Backpropagation(training,FILE_spambase_data)
    backpropagation_acc = backpropagation.data_file()

    """LogisticRegression"""
    print("############LogisticRegression############")
    lRalgo = LogisticRegressionAlgo(training)
    logregression_acc = lRalgo.result()
    print("-------------------------------------------")
    print("")

    """RandomForest"""
    print("############RandomForest############")
    randomForest = RandomForest(training)
    randomforest_acc = randomForest.result()
    print("-------------------------------------------")
    print("")

    """NaiveBayes"""
    print("############NaiveBayes############")
    naiveBayes = NaiveBayes(clean_data.getDf(), clean_data.getNumpy_array(), 0.20)  # 20% de predict
    naivebayes_acc = naiveBayes.result()
    print("-------------------------------------------")
    print("")

    """Knn"""
    print("############Knn############")
    knn = Knn(clean_data.getDf(), clean_data.getNames(), 0.20, 5)
    knn_acc = knn.standardisation()
    print("-------------------------------------------")
    print("")

    """Kernel SVM"""
    print("############Kernel SVM############")
    # Kernel SVM seperate classes by a frontier in a dimension with Kernel method
    # linear trace straight frontier
    # rbf trace straight frontier but accept errors
    kernelSVM = KernelSVM(clean_data.getDf(), clean_data.getNames(), 0.20)
    svm_acc = kernelSVM.results("linear")  # you can set at "linear" or "rbf"
    print("-------------------------------------------")
    print("")

    """Tsne"""
    print("############Tsne############")
    #tsne = Tsne(load_data, FILE_img)
    #tsne.result()
    print("-------------------------------------------")
    print("")

    print("Histogramme")
    accuracy = [backpropagation_acc, logregression_acc, randomforest_acc, naivebayes_acc, knn_acc, svm_acc]
    accuracy_list = pd.Series.from_array(accuracy)
    accuracy_labels = ["Backpropagation", "LogisticRegression", "RandomForest", "NaiveBayes", "Knn", "Kernel SVM"]

    plt.figure(figsize=(12, 8))
    ax = accuracy_list.plot(kind='bar')
    ax.set_title('Accuracy des algos utilisés')
    ax.set_xlabel('Noms des algos')
    ax.set_ylabel('Accuracy')
    ax.set_xticklabels(accuracy_labels)


    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    rects = ax.patches


    for rect in rects:
         # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

    plt.show()
