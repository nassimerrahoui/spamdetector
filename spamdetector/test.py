# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pprint

# Loading dataset
iris_df = datasets.load_iris()
pprint.pprint(iris_df)
print(type(iris_df.data))
print(type(iris_df.feature_names))
print(type(iris_df.target))
print(type(iris_df.target_names))
print(type(iris_df))

print(len(iris_df.target))
print(len(iris_df.data))


# Defining Model
model = TSNE(learning_rate=100)

# Fitting Model
transformed = model.fit_transform(iris_df.data)

# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()