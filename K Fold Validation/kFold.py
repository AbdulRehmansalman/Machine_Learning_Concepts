from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
digits = load_digits()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with increased max_iter and scaled data
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

# K-Fold Cross Validation
kf = KFold(n_splits=3)
print(kf)

for train_index, test_index in kf.split(digits.data):
    print(train_index, test_index)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

model1 = get_score(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test)
model2 = get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test)
print("..................")
print(model1)
print("..................")
print(model2)

# Stratified K-Fold Cross Validation
folds = StratifiedKFold()

scores_lr = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(digits.data):
    X_train_fold, X_test_fold = digits.data[train_index], digits.data[test_index]
    y_train_fold, y_test_fold = digits.target[train_index], digits.target[test_index]

    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)

    scores_lr.append(get_score(LogisticRegression(max_iter=500), X_train_fold, X_test_fold, y_train_fold, y_test_fold))
    scores_svm.append(get_score(SVC(), X_train_fold, X_test_fold, y_train_fold, y_test_fold))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train_fold, X_test_fold, y_train_fold, y_test_fold))

print(scores_lr)
print(scores_svm)
print(scores_rf)

# Cross Validation Scores
print(cross_val_score(LogisticRegression(max_iter=500), digits.data, digits.target))
print("...................")
print(cross_val_score(RandomForestClassifier(n_estimators=16), digits.data, digits.target))
print("...................")
print(cross_val_score(SVC(), digits.data, digits.target))

# Exercise: Use the Iris dataset and cross_val_score for model performance
from sklearn.datasets import load_iris
iris = load_iris()
print(cross_val_score(LogisticRegression(max_iter=500), iris.data, iris.target))
print(cross_val_score(RandomForestClassifier(n_estimators=16), iris.data, iris.target))
print(cross_val_score(SVC(), iris.data, iris.target))
