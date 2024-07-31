import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Display the first few rows of the dataframe
print("Initial DataFrame:")
print(df.head())

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
print("\nDataFrame after dropping unnecessary columns:")
print(df.head())

# Separate target and inputs
target = df.Survived
inputs = df.drop(['Survived'], axis='columns')

# To Chekc if Their is Any Na Vlaue
#todo inputs.columns[inputs.isna().any()]
# Encode the 'Sex' column using LabelEncoder
label_encoder = LabelEncoder()
inputs['Sex_encoded'] = label_encoder.fit_transform(inputs['Sex'])

# Display the first few rows of the encoded 'Sex' column
print("\nEncoded 'Sex' column:")
print(inputs[['Sex', 'Sex_encoded']].head(3))

# Drop the original 'Sex' column from the inputs
inputs.drop(['Sex'], axis='columns', inplace=True)

# Handle missing values in the 'Age' column by filling with the mean
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())
print("\nDataFrame after handling missing values:")
print(inputs.head(5))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=0)

# Initialize the Gaussian Naive Bayes model
NB = GaussianNB()

# Train the model
NB.fit(X_train, y_train)

# Make predictions on the test set
predictions = NB.predict(X_test)

# Display the predictions
print("\nPredictions:")
print(predictions)

# Display the accuracy of the model
accuracy = NB.score(X_test, y_test)
print(NB.predict_proba(X_test[:10]))
print("\nModel Accuracy:")
print(accuracy)


