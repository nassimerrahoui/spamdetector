# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pprint
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Tsne:

    def __init__(self, cd, file_png):
        self.clean_data = cd
        self.file_png = file_png + "Tsne"

    def result(self):
        data_df = self.clean_data.data()
        all_data_df = self.clean_data.getSpambase_data()
        target_df = self.clean_data.target()

        # Defining Model
        model = TSNE(learning_rate=100)

        # Fitting Model
        transformed = model.fit_transform(all_data_df)

        # Plotting 2d t-Sne
        x_axis = transformed[:, 0]
        pprint.pprint(x_axis)
        y_axis = transformed[:, 1]
        pprint.pprint(y_axis)

        plt.scatter(x_axis, y_axis, c=target_df)
        #plt.show()
        plt.savefig(self.file_png)

        # Create the visualizer and draw the vectors
        tfidf = TfidfVectorizer()
        docs = tfidf.fit_transform(data_df)

        tsne = TSNEVisualizer()
        tsne.fit(docs, target_df)
        tsne.poof()