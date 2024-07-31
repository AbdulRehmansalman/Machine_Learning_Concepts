import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read data from CSV file
df = pd.read_csv('myWork.csv')

# Create a linear regression object
reg = LinearRegression()

# Fit the linear regression model to the data
reg.fit(df[['area']], df['price'])

# Plotting data points
plt.scatter(df['area'], df['price'], color='red', marker='*')
plt.xlabel('Area')
plt.ylabel('Price (PKR)')
plt.title('Linear Regression')

# Plot the regression line
plt.plot(df['area'], reg.predict(df[['area']]), color='blue')

# Show plot
plt.show()

# Predicting price for a specific area (e.g., 3300)
area_to_predict = [[3300]]
# This method predicts the target variable ('price') for the provided feature variable ('area') using the trained linear regression model.
predicted_price = reg.predict(area_to_predict)
print("Predicted Price:", predicted_price)

# Printing coefficient and intercept
print("Coefficient (m):", reg.coef_)
print("Intercept (b):", reg.intercept_)

# Predicting prices for areas from another file
areas_to_predict = pd.read_csv('areas.csv')
predicted_prices = reg.predict(areas_to_predict)
areas_to_predict['predicted_price'] = predicted_prices
areas_to_predict.to_csv('PricePredicted.csv', index=False)

#todo If the line closely follows the data points, it indicates a good fit, whereas deviations from the data points suggest
# areas where the model may not accurately capture the relationship between the features and the target variable.