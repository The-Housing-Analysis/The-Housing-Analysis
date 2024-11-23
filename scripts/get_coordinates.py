'''
Obtain coordinates for counties
'''

# import libraries
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import time

# complete cleaned dataset
df_final = pd.read_csv('../data/cleaned_df.csv')

# analysis columns for the pricing families
price_columns = [col for col in df_final.columns if col.startswith('$')] + ['Less than $50,000']
grapi_columns = [col for col in df_final.columns if col.startswith('GRAPI')]
smocapi_columns = [col for col in df_final.columns if col.startswith('SMOCAPI')]
info_columns = ['Year', 'Geography', 'Geographic Area Name']
analysis_cols = info_columns + price_columns + grapi_columns + smocapi_columns

# analysis dataset for the pricing families
analysis_df = df_final[analysis_cols]

# take only the county rows (i.e. ignore state totals)
county_df = analysis_df[analysis_df['Geography'].str.startswith('05')]

# create separate state and county columns
county_df[['County', 'State']] = county_df['Geographic Area Name'].apply(lambda row: pd.Series(row.split(', ')))

# ensure year column is integer type
county_df['Year'] = county_df['Year'].astype(int)

# function to obtain coordinates
def obtain_coordinates(df):
    geolocator = Nominatim(user_agent='housing_analysis')
    coordinates = {'Geographic Area Name': [], 'Latitude': [], 'Longitude': []}
    coordinate_errors = []
    for index, row in df.iterrows():
        # print status
        print(f'{(index / df.shape[0]):.2%}')
        
        # try api
        try:
            row_geog = row['Geographic Area Name']
            location = geolocator.geocode(row_geog)
            if (location.latitude is not None) & (location.longitude is not None):
                coordinates['Geographic Area Name'].append(row_geog)
                coordinates['Latitude'].append(location.latitude)
                coordinates['Longitude'].append(location.longitude)
                print(row_geog)
            else:
                coordinates['Geographic Area Name'].append(row_geog)
                coordinates['Latitude'].append(None)
                coordinates['Longitude'].append(None)
                print(row_geog.upper())
        
        # except api overload
        except:
            coordinate_errors.append(row['Geographic Area Name'])
        
        # add pause to not overload the API
        time.sleep(2)
    
    return coordinates, coordinate_errors

# get unique counties
county_uniques = county_df[['Geography', 'Geographic Area Name']].drop_duplicates(subset=['Geography', 'Geographic Area Name'])
county_uniques.reset_index(drop=True, inplace=True)

# run coordinate obtaining function
county_coordinates, coordinate_errors = obtain_coordinates(county_uniques)

# turn results to dataframe
county_coordinates_df = pd.DataFrame(county_coordinates)

# fix error: Carolina Municipio, Puerto Rico
geolocator = Nominatim(user_agent='housing_analysis')
error_location = geolocator.geocode('Carolina, Puerto Rico')
error_latitude = error_location.latitude
error_longitude = error_location.longitude

# error df
error_df = pd.DataFrame({'Geographic Area Name': ['Carolina Municipio, Puerto Rico'], 'Latitude': [error_latitude], 'Longitude': [error_longitude]})

# concatenate fixed error with rest of county coordinate dataframe
county_coordinates_df = pd.concat([county_coordinates_df, error_df])
county_coordinates_df.reset_index(drop=True, inplace=True)

# save final county coordinate dataframe
county_coordinates_df.to_csv('../data/county_coordinates.csv', index=False)
