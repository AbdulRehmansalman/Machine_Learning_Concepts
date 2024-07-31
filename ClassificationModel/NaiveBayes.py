import Pandas1 as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

# Load the dataset from CSV
df = pd.read_csv("Original_data_with_more_rows.csv")

# Encoding categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['ParentEduc'] = le.fit_transform(df['ParentEduc'])
df['LunchType'] = le.fit_transform(df['LunchType'])
df['TestPrep'] = le.fit_transform(df['TestPrep'])
df['EthnicGroup'] = le.fit_transform(df['EthnicGroup'])


# Splitting the dataset into training and testing sets
X = df.drop('TestPrep', axis=1)
y = df['TestPrep'] # Yes or No
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)
print("The Predicted Values is",y_pred)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

