import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn



digits = load_digits()
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# As it is A 2 d Array of Numbers to represent a Number by a Numericals Format and Length of thsi Array is 64
plt.show()

print(digits.data[:4])

df =pd.DataFrame(digits.data)
# Create a New Column in Pandas Dataframe
df['target']= digits.target
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),digits.target, test_size=0.2, random_state=42)
print(len(X_test))
print(len(y_test))

# todo EnsembleTechnique is used to When you are Using Multiple Algo to predfict the Outcome:
# todo Here we are Building Multiple Decision Trees and Taking the Majority to Come Up WithOur Final Outcome
model = RandomForestClassifier(n_estimators=20) #If I Can Increase the Trees So It's Accuracy can be Better
model.fit(X_train, y_train)
# todo;Gini and Entrophy Criteria is Used in DT. NEstimators means it Used __ Treess to Build the Random Forest
print(model.score(X_test, y_test))

# Confusion Matrix Aloows you to Plot truth on one Axis andPredictyion on the Anopther Axis
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visulize the Confusion atrix on to the SeaBornLibrary
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()