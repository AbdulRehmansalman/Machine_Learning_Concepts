from sklearn.cluster  import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# First You Make a Dataframe and Put the Things in it
df = pd.read_csv('income.csv')


plt.scatter(df['Age'],df['Income($)'])
# plt.show()


km = KMeans(n_clusters=3)
print(km)

# Them Fit and Preddict: Exclusign the name Bcoz it is not useful:it RunsAge and Income on thsi Scatter PLot And it Computed the Cluster as er Criteria ---3
# Where we Identify Three Clusters;
y_pred= km.fit_predict(df[['Age','Income($)']])
print(y_pred)

# Now to Visulize the Aray to Better uNDERSTAND: We Separte into Differenet
df['Cluster'] = y_pred
print(df.head())

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='blue')

plt.xlabel('Age')
plt.ylabel('Incone($)')
plt.legend()
plt.show()

# /As the Proble mArise as Our Scaling is Not Right: Our y os Scaled through 160000 and Our X scales Thorough 42.5
# todo We use MinMAXScalar to Scale These Tw oFeatures And Then We Run out Algorithm,Preprocessing
scalar = MinMaxScaler()
scalar.fit(df[['Income($)']])
df['Income($)'] = scalar.transform(df[['Income($)']])
print(df)
# For Age Also
scalar.fit(df[['Age']])
df['Age'] = scalar.transform(df[['Age']])
print(df)

# Now the Features are Scaled Now Do Apply the kMEANS Algo to the Data
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Age','Income($)']])

df['Cluster'] = y_pred
print(df)

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='blue')
# todo To Visulize the Centroids
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='green',marker='*') #todo Go Through All Rows and One Column
plt.xlabel('Age')
plt.ylabel('Incone($)')
plt.legend()
plt.show()

# ToCheck the Centroids
centroids = km.cluster_centers_.astype(int)
print(centroids)
# todo IfYou have many Many Features Tha Graph Becomes Messy So : Use Elbow Plots
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    # Inertia Method Will Give you the Sun SQAUREED error
    sse.append(km.inertia_)
    print(sse)

# Vuisulize it
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng,sse)