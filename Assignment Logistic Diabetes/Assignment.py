import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#todo First Get the Data From the CSV fIle
df = pd.read_csv('diabetes.csv')
print(df)

#todo Then Preprocess the datset By Handling Missing Values
df.replace({'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 0}, np.nan, inplace=True)
# First Replace Missing Values With Nan and Then Fill missing values with mean
df.fillna(df.mean(), inplace=True)

#todo Scale the features to make all Features at Similar Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Outcome', axis=1))
print("TheScaledFeatures are",scaled_features)
print("The Standard Deviation is ",scaled_features.std(axis=0)) # It is To be One
print("The Mean is ", scaled_features.mean(axis=0))

#todo correlation between Glucose and outcome
cd = pd.crosstab(df.Glucose,df.Outcome)
pd.set_option('display.max_columns', None)
print("The Relation Between Glucose and Dibetes is ",cd)
cd.plot(kind='bar')
# plt.show()

# Correlation Between BloodPressure and outcome
cd = pd.crosstab(df.BloodPressure,df.Outcome)
cd.plot(kind='bar')
# plt.show()
Mean = df.groupby('Outcome').mean()
print(Mean)
# to Check its Mean Who has Less and more Chances According to Features

#todo Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Outcome'], test_size=0.2, random_state=42)

#todo Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#todo Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


print("Shape of the training data:", X_train.shape)
print("Shape of the testing data:", X_test.shape)
print("Number of features:", X_train.shape[1])