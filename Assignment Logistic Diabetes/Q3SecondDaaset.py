import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("cardekho_dataset.csv")
print(df)


#todo Preprocess the data by encoding categorical variables and scaling numerical features.
# First Check the Missing Values
null_columns = df.columns[df.isnull().any()]
print("Columns with null values:")
print(null_columns)

if not null_columns.empty:
    # Calculate mean for each column and Fill in the Datset by Fillna Method
    means = df[null_columns].mean()
    df[null_columns] = df[null_columns].fillna(means)
    print("Means for columns with null values:")
    print(means)
else:
    print("No null columns")
print(df.dtypes)
# Encode the Ctaegorical Data into Numeric So to gte More Accrate Result
categoricalCol = ['car_name','brand', 'model','seller_type','fuel_type','transmission_type']

label_encoders = {}
for col in categoricalCol:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])
pd.set_option('display.max_columns', None)
print("The Df is ",df)

# Seperate x(Features) and labels (Y)
X = df.drop(['selling_price', 'car_id'], axis=1)
y = df['selling_price']

# Scale Numerical Features that they are Converted into Same SCALE To contribute Equally in pRocees of Linear Regressio Ml
scale_features=['brand','model','vehicle_age','km_driven','seller_type','fuel_type','transmission_type','engine','mileage','max_power','seats']
Scaler = StandardScaler()
X[scale_features]=Scaler.fit_transform(X[scale_features])
# print(X)

# todo Split the DataSet Into x AND Y tESting and Training Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Pred The Testing Valaue
y_pred = lr.predict(X_test)

# todo Predict the Feature Importnace Analysis
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
top_features = feature_importance.head(9)['Feature'].tolist()
print("The Top 9 Fetures That are affecting Price are" , top_features)

print("The Predicted Values For Seliing Price is ", y_pred)
mse_top = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("R-squared (R2) score:", r2)
print("The mean Squared Error is ", mse_top)
