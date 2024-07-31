import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree


# todo The Implementation is by Krish
# iris = load_iris()
# print(iris.data)
# classifier = DecisionTreeClassifier()
# classifier.fit(iris.data,  iris.target)
#
# # Visualize the decision tree with customized parameters
# plt.figure(figsize=(20,10))
# tree.plot_tree(classifier,
#                filled=True,
#                feature_names=iris.feature_names,
#                class_names=iris.target_names,
#                rounded=True,
#                proportion=True,
#                precision=2)
#
# plt.show()

# todo Code Basics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('salaries.csv')
print(df.head())

# Separate INto Inputs andTarget Column
inputs = df.drop(['salary_more_then_100k'], axis='columns')
print(inputs)
target = df['salary_more_then_100k']
print(target)

# Get the Categorical DATA into Numerical Data
le_job = LabelEncoder()
le_company = LabelEncoder()
le_degree = LabelEncoder()

# Apply the Fit Transform Method on the new Columns
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

print(inputs.head())
#todo  Next Step;Drop Those Label Columns
inputs_n   = inputs.drop(['company','job','degree'], axis='columns')
print(inputs_n)

#todo Now Train the Classifier
model =tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

print(model.score(inputs_n, target))

# to Test the Model if it is Working Fine
print(model.predict([[2,0,1]]))

#todo Visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model,
               filled=True,
               feature_names=['company_n', 'job_n', 'degree_n'],
               class_names=['<=100k', '>100k'],
               rounded=True,
               proportion=True,
               precision=2)
plt.savefig('decision_tree.png')
plt.show()