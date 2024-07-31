import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Read the dataset
data = pd.read_csv('Original_data_with_more_rows.csv')

# Display the first few rows of the dataset
print(data.head())

# Separate features (x) and target variable (y)
x = data.drop('TestPrep', axis=1)  # Assuming 'TestPrep' is the target variable, adjust as needed
y = data['TestPrep']

# Encode categorical variables
categorical_columns = ['Gender', 'EthnicGroup', 'ParentEduc', 'LunchType']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
x_encoded = ct.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.20)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize and train K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train_scaled, y_train)

# Make predictions
y_pred = classifier.predict(x_test_scaled)

# Evaluate the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot test preparation completion for true and predicted values
# Plot
# Plot test preparation completion for true and predicted values
plt.figure(figsize=(8, 6))

# Plotting the count of test preparation completion for true values
sns.countplot(x=y_test, palette=['#1f77b4', '#ff7f0e'], label='True')

# Plotting the count of test preparation completion for predicted values
sns.countplot(x=y_pred, palette=['#1f77b4', '#ff7f0e'], alpha=0.7, label='Predicted')

# Customizing plot labels and title
plt.xlabel('Test Preparation Status')
plt.ylabel('Count')
plt.title('True vs Predicted Test Preparation Completion')

# Adding legend
plt.legend(labels=['True', 'Predicted'])

# Customizing x-axis ticks
plt.xticks(ticks=[0, 1], labels=['Not Completed', 'Completed'])

# Display the plot
plt.show()


