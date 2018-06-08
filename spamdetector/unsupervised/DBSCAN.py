# Importing Modules
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from spamdetector.cleanData.CleanData import CleanData
from spamdetector.cleanData.Training import Training

class DBSCANalgo :
    def __init__(self, cl, file_img):
        self.file_img = file_img
        self.clean_data = cl
        self.dbscan = DBSCAN()

    def result(self):

        data_df = self.clean_data.data()

        self.dbscan.fit(data_df)

        # Transoring Using PCA
        pca = PCA(n_components=2).fit(data_df)
        pca_2d = pca.transform(data_df)

        # Plot based on Class
        for i in range(0, pca_2d.shape[0]):
            if self.dbscan.labels_[i] < 0.3:
                c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
            elif self.dbscan.labels_[i] >= 1:
                c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')

        plt.legend([c1, c2], ['Cluster 1', 'Cluster 2'])
        plt.title('DBSCAN finds 2 clusters and Noise')
        plt.save(self.file_img)
