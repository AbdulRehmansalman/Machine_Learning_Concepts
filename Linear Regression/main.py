import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


# WE Import Dataset of Diabetes Which Tell that the PAtient is Diabetic or not
diabetes = datasets.load_diabetes()
num_features = diabetes.data.shape[1]

print("Number of features:", num_features)
feature_names = diabetes.feature_names
print("Feature names:")
for feature in feature_names:
    print(feature)


# App ne Dekhna ke Dtaset ma kia kia Hai
print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.data)
#An NumPy aRRAYS ajAYE GI k AbNumPy aRRAYS kIA HAI Descr Karan Pare ga
print(diabetes.DESCR)
# ye Feature ka Use Kar rha hai k age body averageBlood Vlaue de Rha Wo Diabetic Kitna hai

#todo For Double Linear Regression Thsi ContainsFeatures t o Predict Target(Labels)
# diabetes_X = diabetes.data
#todo it Only takes Index 2 wala Feature Column ma Daal kar de dia hai For Single Linear Regression
diabetes_X = diabetes.data[:, np.newaxis,2]
print(diabetes_X)

# Start Ke 30 Records le le Ge For Testing
diabetes_X_Train = diabetes_X[:-30]
# For Training
diabetes_X_Test = diabetes_X[-30:]
#todo On Xaxis Features and On Y Axis We have Label We Have Target Variables
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Aun ke Data ko dekh kar
# Linear LLine Fit Krein ge and aus line se anai walai Features ko Predict karein ge Aus ke Linear Model Banana Paree Ga
model = linear_model.LinearRegression()
# Ais ke Baad Data ko FitTrain) Kraoun ga(Matlab ke Apnai Data ki Help se Aik Line banao ga AurWo Line lINEAR MODEL ma save ho jye gi) To Minimize Difference Between x and Y
model.fit(diabetes_X_Train, diabetes_y_train)
#  Uper tak Model Ban Chuka::Traning means that We are Learning Hum Wo Line Dhoondh rhe,and test The Line we Find is Good or Bad,
# Ke Baki features se Sahi Labels A ArHE
# Training Data se Train kar rhe or Test Data se Test Karein ge Then Testing Down Predict se Value  Bta dia hai
diabetes_y_predicted = model.predict(diabetes_X_Test)

# And the Sum of Squard Ko Average kro TU MEAN SQUARED AVERAGE
# First Paramater is DiabetesYTEST AND  diabetesypredicted is the Second Paramter:) Kia Actual hai Or Kia Model ne Predict kai hai
print("Mean squared error:",mean_squared_error(diabetes_y_test , diabetes_y_predicted))

# Weights are Cofficient Before Independent Variable and Intercepts add
print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

# Plot on Graph
plt.scatter(diabetes_X_Test, diabetes_y_test)
# and draw the line:
plt.plot(diabetes_X_Test, diabetes_y_predicted)
plt.show()
# Humarai jo Scattered Point The,Aun keBeech Ma se JA RHE,Meaned Sqaured error ko Minimized KarTe hoa
# Care That Sum ofSquare Eroor Minimu Rhe
# As itGives Only One Weight Because We Fit on mx+b

# Mean squared error: 3035.060115291269
# Weights:  [941.43097333]
# Intercept:  153.39713623331644