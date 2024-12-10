# -*- coding: utf-8 -*-
"""The_Housing_Analysis - EDA - Architecture.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F-ONV0ghUAghVmFNcGrE02oP5LgPVj-F
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import data
df = pd.read_csv('/content/Annual Housing Data.csv', index_col=0)
df = df[['Geographic Area Name','Geography','1 bedroom', '2 bedrooms', '3 bedrooms', '4 bedrooms',
       '5 or more bedrooms', 'No bedroom','1 room',
       '2 rooms', '3 rooms', '4 rooms', '5 rooms', '6 rooms', '7 rooms',
       '8 rooms', '9 rooms or more', '1-unit, attached', '1-unit, detached', '10 to 19 units', '2 units',
       '20 or more units', '3 or 4 units', '5 to 9 units','Boat, RV, van, etc.', 'Mobile home','Built 1939 or earlier', 'Built 1940 to 1949', 'Built 1950 to 1959',
       'Built 1960 to 1969', 'Built 1970 to 1979', 'Built 1980 to 1989',
       'Built 1990 to 1999', 'Built 2000 to 2009', 'Built 2010 or later',
       'Built 2010 to 2013', 'Built 2010 to 2019', 'Built 2014 or later',
       'Built 2020 or later', 'Year']]

df["Geography Level"] = df['Geography'].apply(lambda x: 'Country' if x[0:2] == "01" else
                                              'State' if x[0:2] == "04" else 'County')
df

df["Total Units in Bedroom Domain"] = df['1 bedroom'] + df['2 bedrooms'] + df['3 bedrooms'] + df['4 bedrooms'] + df['5 or more bedrooms'] + df['No bedroom']
df["Total Units in Room Domain"] = df['1 room'] + df['2 rooms'] + df['3 rooms'] + df['4 rooms'] + df['5 rooms'] + df['6 rooms'] + df['7 rooms'] + df['8 rooms'] + df['9 rooms or more']
df["Total Units in Structure Domain"] = df['1-unit, attached'] + df['1-unit, detached']+ df['10 to 19 units'] + df['2 units'] + df['20 or more units'] + df['3 or 4 units'] + df['5 to 9 units']+df['Boat, RV, van, etc.'] +df['Mobile home']
df["Total Units in Years Built Domain"] = df['Built 1939 or earlier'] + df['Built 1940 to 1949'] + df['Built 1950 to 1959'] + df['Built 1960 to 1969'] + df['Built 1970 to 1979'] + df['Built 1980 to 1989'] + df['Built 1990 to 1999'] + df['Built 2000 to 2009'] + df['Built 2010 or later'] + df['Built 2010 to 2013'] + df['Built 2010 to 2019'] + df['Built 2014 or later'] + df['Built 2020 or later']

bedroom_cols = ['1 bedroom', '2 bedrooms', '3 bedrooms', '4 bedrooms',
       '5 or more bedrooms', 'No bedroom']
room_cols = ['1 room', '2 rooms', '3 rooms', '4 rooms', '5 rooms', '6 rooms',
       '7 rooms', '8 rooms', '9 rooms or more']
structure_cols = ['1-unit, attached', '1-unit, detached', '10 to 19 units', '2 units',
       '20 or more units', '3 or 4 units', '5 to 9 units','Boat, RV, van, etc.', 'Mobile home']
year_cols = ['Built 1939 or earlier', 'Built 1940 to 1949', 'Built 1950 to 1959',
       'Built 1960 to 1969', 'Built 1970 to 1979', 'Built 1980 to 1989',
       'Built 1990 to 1999', 'Built 2000 to 2009', 'Built 2010 or later',
       'Built 2010 to 2013', 'Built 2010 to 2019', 'Built 2014 or later',
       'Built 2020 or later']
df["Avg Bedrooms"] = (df["1 bedroom"]*1 + df["2 bedrooms"]*2 + df["3 bedrooms"]*3 + df["4 bedrooms"]*4 + df["5 or more bedrooms"]*5) / df["Total Units in Bedroom Domain"]
df["Avg Rooms"] = (df['1 room']*1 + df['2 rooms']*2 + df['3 rooms']*3 + df['4 rooms']*4 + df['5 rooms']*5 + df['6 rooms']*6 + df['7 rooms']*7 + df['8 rooms']*8 + df['9 rooms or more']*9) / df["Total Units in Room Domain"]

for col in bedroom_cols:
  df[col] = df[col] / df["Total Units in Bedroom Domain"]
#  df[col] = df[col].apply(lambda x: round(x, 2)*100)

for col in room_cols:
    df[col] = df[col] / df["Total Units in Room Domain"]
#    df[col] = df[col].apply(lambda x: round(x, 2)*100)

for col in structure_cols:
    df[col] = df[col] / df["Total Units in Structure Domain"]
#    df[col] = df[col].apply(lambda x: round(x, 2)*100)

for col in year_cols:
    df[col] = df[col] / df["Total Units in Years Built Domain"]
#    df[col] = df[col].apply(lambda x: round(x, 2)*100)

countryDF = df[df["Geography Level"] == "Country"]
#Drop Country
df = df[df["Geography Level"] != "Country"]
#If county, add add state
df["State Name"] = df.apply(lambda x: x["Geographic Area Name"] if x["Geography Level"] == "Country" else
                                      x["Geographic Area Name"] if x["Geography Level"] == "State" else
                                      x["Geographic Area Name"].split(", ")[1] if x["Geography Level"] == "County" else
                                      x["Geographic Area Name"], axis = 1)

df

#Drop puerto rico
df = df[df["State Name"] != "Puerto Rico"]

#Merge in region data to metroData
regions = pd.read_csv('https://raw.githubusercontent.com/cphalpert/census-regions/master/us%20census%20bureau%20regions%20and%20divisions.csv')
regions = pd.DataFrame(regions)
#regions["State"] = regions["State"].str.lower()
regions

#Merge regions on state names
df = df.merge(regions, left_on='State Name', right_on='State')
df = df.drop(["State Code", "Division"], axis = 1)
df

countiesDF = df[df["Geography Level"] == "County"]
stateDF = df[df["Geography Level"] == "State"]

stateDF2023 = stateDF[stateDF["Year"] == 2023]
sns.scatterplot(data = stateDF2023, x = 'Avg Rooms', y = 'Avg Bedrooms', hue = 'Region')
plt.title("Average Rooms vs Average Bedrooms by State - 2023")
plt.annotate("District of Columbia", xy = [4.4, 1.9])
plt.annotate("Utah", xy = [6.00,3.27])

plt.show()

#Scatter plot of average # of rooms vs average # of bedrooms by county
#Color by region
countiesDF2023 = countiesDF[countiesDF["Year"] == 2023]
sns.scatterplot(data = countiesDF2023, x = 'Avg Rooms', y = 'Avg Bedrooms', hue = 'Region')
plt.title("Average Rooms vs Average Bedrooms by County - 2023")
#Annotate low data point
plt.annotate("New York City, NY", xy = [3.47,1.50])
plt.show()

#Stacked Bars of Years built by state

countryDF
countryDF['Built 2010 to 2019'] = countryDF['Built 2010 or later'] + countryDF['Built 2010 to 2013'] + countryDF['Built 2014 or later'] + countryDF['Built 2010 to 2019']

countryDF = countryDF[['Built 1939 or earlier', 'Built 1940 to 1949', 'Built 1950 to 1959',
       'Built 1960 to 1969', 'Built 1970 to 1979', 'Built 1980 to 1989',
       'Built 1990 to 1999', 'Built 2000 to 2009', 'Built 2010 to 2019',
       'Built 2020 or later', 'Year']]
countryDF

#Filter by year
countryDF2023 = countryDF[countryDF["Year"] == 2023].drop("Year", axis = 1)
countryDF2018 = countryDF[countryDF["Year"] == 2018].drop("Year", axis = 1)
countryDF2013 = countryDF[countryDF["Year"] == 2013].drop("Year", axis = 1)
#Keep only build years

#Long format for country
countryDF2023long = countryDF2023.melt()
# Drop values of 0
countryDF2023long = countryDF2023long[countryDF2023long["value"] != 0]
countryDF2018long = countryDF2018.melt()
countryDF2018long = countryDF2018long[countryDF2018long["value"] != 0]
countryDF2013long = countryDF2013.melt()
countryDF2013long = countryDF2013long[countryDF2013long["value"] != 0]

countryDF2023long

#Three pie charts of 2013, 2018, and 2023 US housing unit data
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(311)
ax1.pie(countryDF2013long["value"], labels = countryDF2013long["variable"], autopct='%1.1f%%')
ax1.set_title('2013')
ax2 = fig.add_subplot(312)
ax2.pie(countryDF2018long["value"], labels = countryDF2018long["variable"], autopct='%1.1f%%')
ax2.set_title('2018')
ax3 = fig.add_subplot(313)
ax3.pie(countryDF2023long["value"], labels = countryDF2023long["variable"], autopct='%1.1f%%')
ax3.set_title('2023')

stateDF

stateDF = stateDF[['State Name','Region','Year','1-unit, detached', '1-unit, attached',  '2 units',
      '3 or 4 units', '5 to 9 units','10 to 19 units',  '20 or more units','Boat, RV, van, etc.', 'Mobile home']]
stateDF

stateDF2023 = stateDF[stateDF["Year"] == 2023]
stateDF2018 = stateDF[stateDF["Year"] == 2018]
stateDF2013 = stateDF[stateDF["Year"] == 2013]

stateDF2023 = stateDF2023.drop(["Year"], axis = 1)

#stacked bar plot of data by state
stateDF2023.plot(x = 'State Name', kind = 'bar', stacked = True, figsize = (15,5))
#Rotate x labels
plt.xticks(rotation=90)
plt.legend(["1-unit, detached","1-unit, attached",  "2 units", "3 or 4 units", "5 to 9 units", "10 to 19 units", "20 or more units", "Boat, RV, van, etc.", "Mobile home"])
plt.title("Housing Units by State - 2023")
plt.ylabel("Percent of Housing Units")
plt.show()

