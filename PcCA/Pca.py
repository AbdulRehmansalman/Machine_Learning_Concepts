import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

dataset = load_digits()
# print(dataset.shape())

print(dataset.data[0])

# /AsEach Sampel has 64 Samples So Convert thhe One D aRRAY TO 2D array
dataset.data[0].reshape(8,8)

# to Visulize the Data
plt.gray()
plt.imshow(dataset.data[0].reshape(8,8))
# plt.show()

# As to sEE Unique Datsets From the
np.unique(dataset.target)

# to Get that Data in the DatFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())

X= df
Y= dataset.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X_scaled , Y, test_size=0.2, random_state=42)

# We Can Go With Differemnt Classification Techniques as Random Forest,Decision Tress
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(train_X, train_y)
print(classifier.score(test_X, test_y))


# Them Do PcA
from sklearn.decomposition import PCA
PCA = PCA(n_components=0.95) #todo it Means that the Give me 95% of the UseFul Features
X_Pca = PCA.fit_transform(X)
print(X_Pca.shape) # 29 Columns are More Important than 64 Columns Else We uSE all 64 Columns;It Will Not Take 29 Columns it Will Calculate new Columns

print(PCA.explained_variance_ratio_)
print(PCA.n_components)

# So New DatFrame to Train the Model Which we have 29 Cols
X_Train_Pca,X_Test_pca,y_train,y_test =train_test_split(X_Pca,Y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_Train_Pca,y_train)
print(model.score(X_Test_pca,y_test))

# /You can define by define componets Expliculy say Componets are 2
pca1 = PCA(n_components=2)
x_Pca=pca1.fit_transform(X)
print(x_Pca)

# As THEN call the model as It Will Get the Coputational So Fast but Reduce the Accuracy
