import Pandas1 as pd
import numpy as np

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}

df = pd.DataFrame(data)
print (df)

df.to_csv('data2.csv')
# If we Dont WANT T OWrite Indexes
df.to_csv('data.csv',index=False)

print (df)
print("....................")
# to get The Above 2 Records
print(df.head(2))
# to Print the Last 2 Records
print(df.tail(2))
# to Fidn the Count Mean Standand Deviation min max 25%
print(df.describe())
print("....................")

# To Read Data From the Csv
abdurrehman = pd.read_csv('data2.csv')

# Display the DataFrame
print(abdurrehman)
# Change the value of the 'Age' column in the first row to 600
abdurrehman.loc[0, 'Age'] = 600
# Display the modified DataFrame After Modifying the Data2.csvFile Then we Again Retain its ModifiedVlaue
abdurrehman.to_csv('data2.csv')

# To Chnage the Index To Another Exaple One Two Three
abdurrehman.index = ['First', 'Second', 'Third', 'Fourth']
print(abdurrehman)
