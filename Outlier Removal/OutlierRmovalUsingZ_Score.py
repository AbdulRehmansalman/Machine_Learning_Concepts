import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (10,6)
df = pd.read_csv('heights.csv')
print(df.sample(5))


# AbBellShaped Curve mean that the majority of the values are centered around mean and on average;And as ou go Awayfrom the Mean the Number of Vlaues Goes Down.
plt.hist(df.height,bins=20,rwidth=0.8)
plt.xlabel('Height(inches)')
plt.ylabel('Count')
# plt.show()

# todo to Show the BellCurve:
from scipy.stats import norm
plt.hist(df.height,bins=20,rwidth=0.8)
plt.xlabel('Height(inches)')
plt.ylabel('Count')
rng =np.arange(df.height.min(),df.height.max(),0.1)
plt.plot(rng,norm.pdf(rng,df.height.mean(),df.height.std()))
plt.show()

# Standard Deviation SHow How Far Your Dat is Away from the MEAN Vlaue
# As 66 + 3.84 is One aTandard Deviation Away
# And -3.84 is 2 Staandard Deviation Away
# We Use 3 Satndard Deviation For Removal of the Outliers 4or 5 if small 2 Use:
# todo Use three Standard Deviation for the REMOVAL of the Outlier
upper_Limit = df.height.mean() + 3*df.height.std()
print(upper_Limit)  #Above it Mark as THE oUTLIER

lower_limit = df.height.mean() - 3*df.height.std()
print(lower_limit)  #Less than it is Marked as Outlier

# to Chekc EWhic is Outlier
print(df[(df.height > upper_Limit) | (df.height<lower_limit)])
print(df.shape)

df_no_outlier_std = df[(df.height < upper_Limit) & (df.height > lower_limit)]
print(df_no_outlier_std.shape)

print(df.shape[0] - df_no_outlier_std.shape[0])



