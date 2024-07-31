import pandas as pd
# Dataframe is the main Object Object in padas andIt is Used to reresent Row and Columsn and (TABULAR OeExcel Sheet )
df = pd.read_csv('weather_data.csv')
print (df)
# It Define the Diensions of the Dataset Row sand Columsns
print (df.shape)
rows,Columns = df.shape
 
# The Record Between 2 Till 3 Nor Include 4
print("The Data of the Record 2 till 4 ",df[2:5]) 

print("The Days are : /n",df['day'])


# It Defines all the data As it Contains Describe()
print(df.describe())
print(df[df['temperature'] == df['temperature'].max()])

# Set Index Chnages the Indexes of the DatFrame
# Inplace = True Will Make the Chnage in the Original Df Else if not It Only Chnages to New Df
df.set_index('day',inplace=True)
print(df)

#? LocMethod to get the Row of the DatAfRAME
print("The Location at 1/6/2017 is: ",df.loc['1/6/2017'])

# todo If to Replace Indexes
df.reset_index(inplace=True)
print("The reRest Index is Now",df)

# Put Events: 
df.set_index('event',inplace=True)
print("THe Events are ",df)

print("The Snow Locations are ",df.loc['Snow'])

# todo Creating Python Dictionary UsING DatFrame ()

waethe_data = {
    'day': ['1/1/2019','1/2/1223','2/3/4454'],
    'temperature': [32,35,36],
    'windspeed': [6,7,6]
}
print("....................")
df1 = pd.DataFrame(waethe_data)
print(df1) 
print("....................")
# todo Creating DataFram USing Tuple List 
weather_data=[
    ('1/1/2019',32,6,'Rain'),
    ('1/2/1223',7,9,'Moderate'),('2/3/4454',7,9,'Sunny')
] 
df2 = pd.DataFrame(weather_data,columns=["day", "Temperature", "WindSpeed","Event"])
print (df2)

# *Reading Through Csv 
print("Whenever You have Missing Header in the CSv File Then USe ") 
print("....................") #todo If Your Data is Large and Yoou Raed Only 3 RowesEtch:: nrows=3

#? df3 = pd.readcsv("weather_data",header=None ,names=[] skiprows=1, nrows=3)
#! Replace Not Available oR N.a. vLAUE into Nan:
#? df3 = pd.readcsv("weather_data",na_values=["Not Available","n.a."])

# todo If You Wnat to Chnage the Value that is Negative Than You Want Only One Clumn To ChnageIn so Intead of List USE dictionaries
# ? df3 = pd.readcsv("weather_data", na_values='eps': ["Not Available","n.a."],'revenue':["Not Available","n.a."])


#! Writing Through Csv:You can Also Select ColuMn Name to Write in csv 
df.to_csv('new_csv.csv',columns=['tickers','eps']) # And We Also Remove the Headers by headers=Flase

#? To Reaf Data From the Excel
# pp = pd.read_excel('new_csv.csv');

