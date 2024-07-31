# todo z Score Gives you the Number And Tells How many Standard Deviation Away From the Mean
# todo Exmaple: if Your MEAN IS 66.37 and Your Standard Deviaton is 3.84
# todo if the DatapoInt is 77.89 then z_score is 3, (77.91 = 66.37+ 3* 3.84)

import pandas as pd

df = pd.read_csv('heights.csv')
# to Find the Z_score
df['z_score'] = (df.height - df.height.mean()) / df.height.std()
print(df.head(5))

print(df.height.std())
print(df.height.mean())
# print(5.9 - 6.05/2.779) is Sme as in Df Cumn

# todo to Detct Outliers
print(df[df['z_score']>3])
print(df[df['z_score']<-3])

# to remove These Outlers You get IntoAnother DataFrame
df_no_outliers = df[(df.z_score >-3) & (df.z_score<3)]
print(df_no_outliers.head(5))