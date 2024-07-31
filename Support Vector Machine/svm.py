import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
# Target is detemined by these Four Above Features
df['target'] = iris.target # ItContain Three Flowers
print(df.head())


# To Chekc Which Flower is Which And What is BNest FLOWER
# todo So 1-50 is OneFlower 51-100 is the Next Flower and 101-150 is Next Flower

print(df[df.target==1].head())

# todo Add One more Coloumn as Flower name :Form One Cloumn You are Trying toGenerate Anothe rColumn
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())


# Visulaizxe the data
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='orange',marker='+')
plt.show()

# Petal Lenght Width F3eATURES
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='+')
plt.show()

# todo When we Train Our Model than,wE tAKE aLL fOUR Features,AndTHree Species
# First Use Test Train Split to SApliut our Model to test and Train Datset
X = df.drop(['target','flower_name'],axis='columns')
print("The X is ",X.head())

y = df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(len(X_test))
print(len(X_train))

# Now Train the Model
# todo Increasing Reulization Will Decvrese My Score
model = SVC(C=10)
model.fit(X_train,y_train)
print(model.predict([[10.1,3.2,1.9,3.2]]))
print(model.score(X_test,y_test))

