import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
print(dir(digits))

print(digits.data[0]) #to Plot the Numerical Format of the  data

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

# As We Take Data and the TARGET TO train the model
print(digits.target[0:5])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target ,test_size=0.2)
print(len(y_train))

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
# Practice

plt.matshow(digits.images[67])
print(digits.target[67])


print(model.predict([digits.data[67]]))

# Confusion Matrtix
y_Pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_Pred) #to Better Visuoize the Data
print(cm)

# Use sEABorn Library to Visulize
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()