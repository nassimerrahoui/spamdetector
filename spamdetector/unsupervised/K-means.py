# Importing Modules
from sklearn import datasets
from sklearn.cluster import KMeans


class K_means:
    def __init__(self, tr):
        # Loading dataset
        iris_df = datasets.load_iris()

        self.tr = tr

    def result(self):
        # Declaring Model
        model = KMeans(n_clusters=3)

        data = self.tr.getX
        # Fitting Model
        model.fit(self.iris_df.data)

        # Predicitng a single input
        predicted_label = model.predict(self.iris_df.data[0])

        # Prediction on the entire data
        all_predictions = model.predict(self.iris_df.data)

        # Printing Predictions
        print(predicted_label)
        print(all_predictions)
        return predicted_label, all_predictions
