import numpy as np
import Pandas1 as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

df = pd.read_csv('data.csv')
# print(df.head())

# The Way i am Handling the Nan Bedroom by Taking the Median

median_bedrooms = math.floor(df.bedrooms.median())
# print(median_bedroom)
# And by Fillna Function We canFil all the Nan Values With Median

df.bedrooms= df.bedrooms.fillna(median_bedrooms)

# print(df) So Our Data Preprocessing Step is Over To CLEAN THE DATA,Prepare then Apply

reg = linear_model.LinearRegression()
# todo To Train the Model Using the Training Set and Now After Mu Model is Ready
reg.fit(df[['area','bedrooms','age']],df.price)

# Now we Have Cofficient as m1,m2,m3
print("The Cofficienrt is ",reg.coef_)
print("The Intercept is ",reg.intercept_)
# After that To pUt Vlaues For PriceTarget Varable
# 300 SQfIR aREA 3 Bedrooms 40 Years Old ,, 2500 SqFoot 4 Bedrroms5Years Old
Predict_price =  reg.predict([[3200,3,45]])
print("ThePredictedPrice is ",Predict_price)
print(df)

# This can beDone By
# 137.25*3000 + -26025*3 + -6825*40 + 383724.9999999998 PREDICTION OF THE PRICE OF HOME
