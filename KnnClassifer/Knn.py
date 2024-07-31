import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = load_iris()

print(iris.feature_names)
print()
print(iris.target_names)

# Load Data into dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
df['target'] = iris.target  # Add 'target' column
df['flower_name'] = iris.target_names[iris.target]  # Add 'flower_name' column

# Put them into DataFrames
df1 = df[:50]
df2 = df[50:100]
df3 = df[100:]

# Plot it
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color="blue", marker='o', label='Versicolor')
plt.scatter(df3['sepal length (cm)'], df3['sepal width (cm)'], color="red", marker='*', label='Virginica')
plt.legend()
plt.show()

# Get into Train and Test Dataset
X = df.drop(['target', 'flower_name'], axis=1)
# print("X is " ,X)
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
# As it Involve Eucladian Distance
print(knn.score(X_test,Y_test))

# Confusion Matrixc to Tells that Which Classes It got the Prediction Right and Wrong
y_pred = knn.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
print(cm)

# todo Anu=yThinf on the Diagonal are the Correct Predictions:11 Times the Flowe is Satosa at 0
# todo as in Second as it is VersiColr But it Says 2 Vergicolor
# Plot This Confsion Matrix
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
# AnD Show Classification Report
print(classification_report(Y_test, y_pred))
