import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('diabetes.csv')
print(df.head())

# To Chekc if there is Any Null Valaues:
print(df.isnull().sum())

print(df.Outcome.value_counts())
# print(500/268) Very Slight Imbalance 2 to 1 Ratio


X = df.drop('Outcome', axis='columns')
y = df['Outcome']

# Scaled the Vlaues:
scaler = MinMaxScaler()
X_scales = scaler.fit_transform(X)
print(X_scales[:3])
# Startify means that it has Equal Ratio: X has EQUAL; rATIO TO Y
X_train, X_test, y_train, y_test = train_test_split(X_scales, y,stratify=y , random_state=10)
# Yout can Check if it Works as Same Ratio as Avbove
print(y_train.value_counts()) #As it is Same as Above Ratio So it Fir ropertly

dt = DecisionTreeClassifier()
scores =cross_val_score(dt, X, y, cv=5)
print(scores.mean())
# ItSays that Yout Model is Making 71 of Accuracy
# Todo Use Bagging Calssifier


# oob Means out of Bag , You have 20 Samples that is not Appering in Any of the Subset and You can Use Those 20 Samples to Train For Prediction,,Take the Majority Vote and Figure out What Was the Accuracy cann OOB Score

bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(),
                  n_estimators=100,
                  max_samples=80,
                  oob_score=True,
                  random_state=0)
bag_model.fit(X_train, y_train)
print(bag_model.oob_score_)
scores =cross_val_score(bag_model, X, y, cv=5)
print(scores.mean())

RandomForest = RandomForestClassifier()
scores = cross_val_score(RandomForest ,X,y,cv=5)
print(scores.mean())
