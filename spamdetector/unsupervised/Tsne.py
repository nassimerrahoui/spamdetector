# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pprint

# TODO: mieux cleaner la data !!! ( il prend bcp de temps le résultat n'est pas super :/ )
class Tsne:

    def __init__(self, ld, file_png):
        self.load_data = ld
        self.file_png = file_png + "Tsne"

    def result(self):
        data_df = self.load_data.data()
        target_df = self.load_data.target()

        # Defining Model
        model = TSNE(learning_rate=100)

        # Fitting Model
        transformed = model.fit_transform(data_df)

        # Plotting 2d t-Sne
        x_axis = transformed[:, 0]
        y_axis = transformed[:, 1]

        plt.scatter(x_axis, y_axis, c=target_df)
        plt.show()
        plt.savefig(self.file_png)