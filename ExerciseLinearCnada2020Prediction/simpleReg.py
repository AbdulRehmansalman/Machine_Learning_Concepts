import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# First Step is to Gather Data
data = pd.read_csv('canada_per_capita_income.csv')
# print(data.head())
#todo Create a Linear Model
reg= linear_model.LinearRegression()
# It fits the model to the training data, meaning it finds the best-fitting line or curve that represents the relationship between the features and the target variable
# todo .Fir Means That You are Tarining the Data by Our Given Data Points
reg.fit(data[['year']],data['per capita income (US$)'])

#todo Plotting data points
plt.scatter(data['year'], data['per capita income (US$)'], color='red', marker='*')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.title('Linear Regression')
# todo to Draw the Regression Line Visually Acces How Well your Linear Reressio Model Fits the Data
plt.plot(data['year'], reg.predict(data[['year']]), color='blue')
# If 'per capita income (US$)' is your target variable and you want to plot a regression line for it based on some other predictor variable(s)
plt.show()

print("The Cofficient m is ",reg.coef_)
print("The Intercept is B ",reg.intercept_)

# After The Model is Trained Then we Want to Predict the data Testing
year_priceto_predict = [[2020]]
predicted_price = reg.predict(year_priceto_predict)
print("The Predicted Price is", predicted_price)

# Calculate Mean Squared Error
mse = mean_squared_error(data['per capita income (US$)'], reg.predict(data[['year']]))
print("Mean Squared Error:", mse)

# todo Updated Verion
import numpy as np
import matplotlib.pyplot as plt
import Pandas1 as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# First Step is to Gather Data
data = pd.read_csv('canada_per_capita_income.csv')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['year']], data['per capita income (US$)'], test_size=0.2, random_state=42)

# Create a Linear Model
reg = linear_model.LinearRegression()

# Fit the model to the training data
reg.fit(X_train, y_train)

# Plotting data points
plt.scatter(X_train, y_train, color='red', marker='*', label='Training Data')
plt.scatter(X_test, y_test, color='blue', marker='o', label='Testing Data')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Linear Regression')
# Plot the regression line for training data
plt.plot(X_train, reg.predict(X_train), color='green', label='Training Line')
# Plot the regression line for testing data
plt.plot(X_test, reg.predict(X_test), color='orange', label='Testing Line')
plt.legend()
plt.show()

print("The Coefficient m is ", reg.coef_)
print("The Intercept is b ", reg.intercept_)

# Predicting the data using testing set
# predicted_prices = reg.predict(X_test)
predicted_prices = reg.predict([[2020]])
print("Predicted Prices for Testing Data:", predicted_prices)

# Calculate Mean Squared Error for testing set
mse_test = mean_squared_error(y_test, predicted_prices)
print("Mean Squared Error for Testing Data:", mse_test)
