import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Melbourne_housing_FULL.csv')
print(df.head())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df.nunique())

# to Check only the Whcih Cols are to Use
col_to_use = ['Suburb', 'Rooms', 'Type','Method','SellerG','Regionname','Propertycount','Distance','CouncilArea','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Price']
df = df[col_to_use]

# To ChekcWhcich Column has How many Na Valaues
print(df.isna().sum())
# We Fill the Coumns to Zero
cols_to_Zero = ['Propertycount','Distance','Bedroom2','Bathroom','Car']
df[cols_to_Zero]=df[cols_to_Zero].fillna(0)


# And Place the Mean of the LandSize and BuildingArea :
df['BuildingArea']=df['BuildingArea'].fillna(df['BuildingArea'].mean())
df['Landsize']=df['Landsize'].fillna(df['Landsize'].mean())
print(df.isna().sum())

# Drop Concil Area and Region name bcoz it is Not Impacting Anuthing
df.dropna(inplace=True)
print(df.isna().sum())

# Get Dummies to Get Numerical Vlaues of theCategoricAL vARIABLES
df = pd.get_dummies(df,drop_first=True) #todo Every Separate Columns for the Column Data
print(df.head())

X = df.drop(['Price'] , axis = 1)
Y= df['Price']

Train_X,Test_X,train_y,test_y =train_test_split(X,Y,test_size=0.2,random_state=2)

lr = LinearRegression().fit(Train_X,train_y)
print(lr.score(Test_X,test_y))
print(lr.score(Train_X,train_y))


# As L1 regilization is Preventing the Model From Overfitting Cndition
from sklearn import linear_model

reg = linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
reg.fit(Train_X,train_y)

print(reg.score(Test_X,test_y))
print(reg.score(Train_X,train_y))

# /From Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=50,max_iter=100,tol=0.1)
ridge.fit(Train_X,train_y)
print("The Ridge L2 Reulization is  ",ridge.score(Test_X,test_y))


