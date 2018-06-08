# Importing Modules
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd


class Classification_hiérarchique:
    def __init__(self, tr, cleanData, file_img):
        self.tr = tr
        self.cleanData = cleanData
        self.file_img = file_img+"Classification_hiérarchique"

    def result(self):
        data_df = self.cleanData.getSpambase_data()
        # Remove the grain species from the DataFrame, save for later
        varieties = list(data_df.pop('is_spam'))

        # Extract the measurements as a NumPy array
        samples = data_df.values

        """
        Perform hierarchical clustering on samples using the
        linkage() function with the method='complete' keyword argument.
        Assign the result to mergings.
        """
        mergings = linkage(samples, method='complete')

        """
        Plot a dendrogram using the dendrogram() function on mergings,
        specifying the keyword arguments labels=varieties, leaf_rotation=90,
        and leaf_font_size=6.
        """
        dendrogram(mergings,
                   labels=varieties,
                   leaf_rotation=90,
                   leaf_font_size=6,
                   )
        plt.save(self.file_img)
