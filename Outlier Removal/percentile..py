import pandas as pd

# df = pd.read_csv('heights.csv')
# print(df.head())
#
# print(df['height'].quantile(0.95))
#
# max_threshold =df['height'].quantile(0.95)
# # Any thing abov this is Considered as Outlier 9.45
# print([df[df['height']<max_threshold]])
# # we can Detct A;os Minimum Data Outlier
#
# min_threshold = df['height'].quantile(0.05)
# print([df[df['height']<min_threshold]])


# So Print THose Which has No Outlier
# print(df[df['height'] >  min_threshold] & df[df['height'] < max_threshold])

# todo For the REAL dATA:
dataset = pd.read_csv("bhp.csv")
print(dataset.shape)
print(dataset.describe()) #Max-Price is not the True Vlaue it is Like the Oulie Which has Very Big Vlaue That is Not Possible Seems to be Outlier
min_threshold,max_threshold = dataset.price_per_sqft.quantile([0.001,0.999])
print(min_threshold," ",max_threshold)

# to See the DataPoints Which have More Values tha nMAX_threshold Then:
print(dataset[dataset.price_per_sqft < min_threshold])
# And mINIMUM
print(dataset[dataset.price_per_sqft > max_threshold])

# To rEMOVE THE outlier You Craete the nEW dATAFrame
df2 =dataset[(dataset.price_per_sqft < max_threshold) & (dataset.price_per_sqft > min_threshold)]
print(df2.shape)

print(df2.sample(10))
