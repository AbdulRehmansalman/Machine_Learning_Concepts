
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('placement.csv')
print(df.shape)


# Plot the Pointsin the Graph
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package in (LPA)')

# : (before the comma) means select all rows.
# 0:1 (after the comma) means select columns from index 0 to 1 (excluding 1). This effectively selects only the first co
X = df.iloc[:,0:1]
y = df.iloc[:,-1]
# : (before the comma) means select all rows.
# -1 (after the comma) means select the last column.

# y = df.iloc[:,0]
# Practice
# Here Split the Data into Test and Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# Apply the Model and Make the Object of it and Train it
lr = LinearRegression()
lr.fit(X_train, y_train)

print(y_test)

# After Traing Prdict the Model Acuracy
# Prediction = lr.predict([[112]])
# Predict=lr.predict(X_test .iloc[1].values.reshape(1,1))
Predict = lr.predict(X_test)
print(Predict)


# Find the Mean Squaredd Errror
mse = mean_squared_error(y_test,Predict)
print("The mean Squared Error is ",mse)
print(lr.score(X_test, y_test))

# Scatter plot of actual vs. predicted values
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.scatter(X_test, Predict, color='blue', label='Predicted')

# Line plot of the model's predictions
plt.plot(X_test, lr.predict(X_test), color='green', label='Regression Line')

plt.title('Actual vs Predicted values')
plt.xlabel('Package')
plt.ylabel('Values')
plt.legend()
plt.show()