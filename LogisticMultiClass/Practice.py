import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


iris = load_iris()

print(iris.keys())

print(iris.target_names)
print(iris.feature_names)

import matplotlib.pyplot as plt

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

X = iris.data
Y = iris.target

print('X Vlaues is ',X)
print("Y VlAUES IS ",Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_pred=lr.predict(X_test)
print(lr.score(X_test, y_pred))

# plt.show()

# To Show the Boundary Line Between it
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
#
# # Load the iris dataset
# iris = load_iris()
# X = iris.data[:, :2]  # we only take the first two features for simplicity
# Y = iris.target
#
# # Split the dataset into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # Fit Logistic Regression
# lr = LogisticRegression()
# lr.fit(X_train, Y_train)
#
# # Create a mesh grid
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#
# # Predict class labels for each point in the mesh
# Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# # Plot the decision boundary
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.title('Logistic Regression Decision Boundary')
#
# # Plot training points
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, marker='o', edgecolor='k')
# # Plot testing points
# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, marker='x', edgecolor='k')
#
# plt.legend(iris.target_names, loc="lower right", title="Classes")
# plt.show()
