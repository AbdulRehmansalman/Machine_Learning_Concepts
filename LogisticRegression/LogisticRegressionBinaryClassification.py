import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('insurance_data.csv')
# print(df.head())
# todo The person That is Younger is less Likely to Buy the Insurenace and More Ages Pele are More Likely
X_train,X_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.2)
print(X_test)
# print(X_train)
# print(y_train)
# print(y_test)

# Train the LogisticRegresson Model
model = LogisticRegression()
model.fit(X_train,y_train)
#Test The Testing Data Point
print(model.predict(X_test))
# todo You can Predict the Probability of X Test
print(model.predict_proba(X_test))

# todo to Check the Score :To Check the Accuracy of the Model
print(model.score(X_test,y_test))
# print(model.predict([[39]]))
# print(model.predict([[40]]))
#todo to plot the Regression Line

X_range = np.linspace(df['age'].min(), df['age'].max(), 100).reshape(-1, 1)
# Predict probabilities for the range of ages
y_range_probs = model.predict_proba(X_range)[:, 1]

plt.scatter(df.age,df.bought_insurance,marker='+',color='purple')
# Plot the Logistic Regressio Line
plt.plot(X_range, y_range_probs, color='red', label='Logistic Regression Line')
plt.show()





