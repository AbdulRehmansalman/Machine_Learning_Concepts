import numpy as np
import matplotlib.pyplot as plt
import Pandas1 as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# Data From Excel Sheet
df = pd.read_csv('HR_comma_sep.csv')
print(df.head())

# Data Exploration and Visulization take Who Left
left = df[df.left ==1]
# left.shape
print(left.shape)

# print(df.dtypes)

retained = df[df.left==0]
print(retained.shape)

#todo Encode Department column using LabelEncoder
label_encoder = LabelEncoder()
df['Department'] = label_encoder.fit_transform(df['Department'])

# Map salary categories to numerical values
df['salary'] = df['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# Get the Mean of All the Columns for each group
pd.set_option('display.max_columns', None)
Mean = df.groupby('left').mean()
print(Mean)
# todo *Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
# **Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
# **Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
# Salary 0.65 No 0.41 Yes

#todo Impact of Salary on Employee Retension
# The crosstab() method returns a DataFrame representing the cross-tabulation of the factors specified in index and columns.
ct= pd.crosstab(df.salary,df.left)
ct.plot(kind='bar')
plt.show() #todo Employee With Good Salary Will Not Leqave the Company

# todo Plot Charts showing corelation With Department and Employees

cd = pd.crosstab(df.Department,df.left)
cd.plot(kind='bar')
plt.show()
# So it is Not makign So Much Impaatc So Ignore Department

#todo So we Conclude tha We Use These Independen Vrable Head Will Sjow First Few Rows of DataFrames
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()

# Salary has All Text Data So Tackle it With Summy Vrable For Readibility
# todo Now Build LogisticRegression Model suing Varibales narrowed Down in Step1 We Found This Independent Variable
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")

# Concatenate the original DataFrame with the dummy variables
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')

# Drop the original 'salary' column
df_with_dummies.drop('salary', axis='columns',inplace=True)
# print(df_with_dummies.head()) # Slary 0 'Low' andso on

# GET X AND Y Data Get DumiesMethod :you convert this categorical variable into binary dummy variables, where each unique category becomes its own binary column.
X = df_with_dummies
print("The X is" ,X)
# X.head() An dY as Who Left

y = df.left
#todo Split the test andTrain DATA
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
# Now MAKE THE MODEL
model = LogisticRegression(max_iter=1000)

# todo Train the Model
model.fit(X_train,y_train)

# todo Predict the Probality of X Testing
y_pred = model.predict(X_test)
print("ThePrediction is" ,y_pred)

# todo To Chekc the ACUURACy
print(model.score(X_test,y_test))