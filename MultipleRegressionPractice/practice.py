import numpy as np
import Pandas1 as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')
# print(df)

# Data Visualize and Refactor The Nan Data in csv File
df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)
# print(df)
median_score = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_score)
print(df)

# Make the Linear Regression Model
model = LinearRegression()

X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = df['salary($)']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
# Train the Model
model.fit(X_train,y_train)

# Then Predict the Data on Given Testing Datset
print("The Y Test is ",y_train)
print("The X Test is ",X_test)
m_predict = model.predict(X_test)
# [51875.7417681]
# m_predIct =model.predict([[12,10,10]])

print("The Predicton is ",m_predict)
print("TheAccuracy is ", model.score(X_test,y_test))

