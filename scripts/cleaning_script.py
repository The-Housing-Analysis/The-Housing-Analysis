'''
Initial Cleaning
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data
data2023 = pd.read_csv('../data/ACSDP1Y2023.DP04-Data.csv')
data2018 = pd.read_csv('../data/ACSDP1Y2018.DP04-Data.csv')
data2013 = pd.read_csv('../data/ACSDP1Y2013.DP04-Data.csv')

# update headers to row0 values
data2023 = data2023.rename(columns=data2023.iloc[0]).drop(data2023.index[0])
data2018 = data2018.rename(columns=data2018.iloc[0]).drop(data2018.index[0])
data2013 = data2013.rename(columns=data2013.iloc[0]).drop(data2013.index[0])

# add year column to each dataset
data2023['Year'] = "2023"
data2018['Year'] = "2018"
data2013['Year'] = "2013"

# reset indices
data2023 = data2023.reset_index(drop=True)
data2018 = data2018.reset_index(drop=True)
data2013 = data2013.reset_index(drop=True)

def clean_column_names(cols):
    cols = [col for col in cols if type(col)==str]
    cols_retain = []
    for col in cols:
        if '!!' not in col:
            cols_retain.append(col)
        elif col.startswith('Estimate!!'):
            cols_retain.append(col)
            
    return cols_retain


'''
def find_different_columns(cols_1, cols_2):
    # items present in first set but not in second set
    return list(set(cols_1).difference(set(cols_2)))

# difference 13 to 18
diff_13_18 = find_different_columns(cleaned_13, cleaned_18)
diff_18_13 = find_different_columns(cleaned_18, cleaned_13)

# difference 13 to 23
diff_13_23 = find_different_columns(cleaned_13, cleaned_23)
diff_23_13 = find_different_columns(cleaned_23, cleaned_13)

# difference 18 to 23
diff_18_23 = find_different_columns(cleaned_18, cleaned_23)
diff_23_18 = find_different_columns(cleaned_23, cleaned_18)
'''

# get YEAR STRUCTURE BUILT appended across all datasets
def get_build_year_info(df):
    build_year_info = []
    for col in df.columns:
        if type(col) == str:
            if 'Estimate!!YEAR STRUCTURE BUILT' in col:
                build_year_info.append(col)
            
    return build_year_info

build_year_info_2013 = set(get_build_year_info(data2013))
build_year_info_2018 = set(get_build_year_info(data2018))
build_year_info_2023 = set(get_build_year_info(data2023))

build_year_info = list(build_year_info_2013.union(build_year_info_2018).union(build_year_info_2023))

# build these columns into each dataset
for build_year in build_year_info:
    if build_year not in data2013.columns:
        print(build_year)
        data2013[build_year] = 0
    if build_year not in data2018.columns:
        print(build_year)
        data2018[build_year] = 0
    if build_year not in data2023.columns:
        print(build_year)
        data2023[build_year] = 0

cleaned_13 = clean_column_names(data2013.columns)
cleaned_18 = clean_column_names(data2018.columns)
cleaned_23 = clean_column_names(data2023.columns)

# columns contained in all
cols_all = list(set(cleaned_13).intersection(set(cleaned_18), set(cleaned_23)))

# select matching columns in datasets
subset_13 = data2013[cols_all]
subset_18 = data2018[cols_all]
subset_23 = data2023[cols_all]

# concat data
df = pd.concat([subset_13, subset_18])
df = pd.concat([df, subset_23])

# finalize columns
# static_columns = ['Year', 'Geography', 'Geographic Area Name']
# df_cols = [col for col in df.columns if col not in static_columns]

# look at tiered column structure
def tier_columns(cols, split_criteria='!!'):
    column_tiers = []
    for col in cols:
        column_tiers.append(col.split(split_criteria))

    return column_tiers

'''
df_tiers = pd.DataFrame(tier_columns(df_cols))
df_tiers.columns = [f'tier_{col}' for col in range(df_tiers.shape[1])]
df_tiers.sort_values(by='tier_1', inplace=True)

first_tier = df_tiers['tier_1'].unique()
second_tier = df_tiers['tier_2'].unique()
third_tier = df_tiers['tier_3'].unique()

This provides:
    - tier_0: Estimate
    - tier_1: 17 options
    - tier_2: 9 options
    - tier_3: 81 options (add 1 for the null values)
    
How should we handle when the third tier values are null? Can it add any values? Does a total make sense for that?
'''

# columns with tier_2 null values to drop
drop_initial = ['BEDROOMS', 'GRAPI', 'HOUSE HEATING FUEL', 'HOUSING OCCUPANCY', 'HOUSING TENURE', 'MORTGAGE STATUS', 'OCCUPANTS PER ROOM', 'ROOMS', 'SMOC', 'SELECTED CHARACTERISTICS', 'SMOCAPI', 'UNITS IN STRUCTURE', 'VALUE', 'VEHICLES AVAILABLE', 'YEAR HOUSEHOLDER MOVED INTO UNIT', 'YEAR STRUCTURE BUILT']
# columns without tier_2 null values to drop
drop_secondary = ['GROSS RENT', 'SMOC', 'YEAR HOUSEHOLDER MOVED INTO UNIT']
# specific columns without tier_2 null values to drop
drop_tertiary = ['ROOMS!!Total!!Median rooms', 'VALUE!!Owner Occupied!!Median (dollars)']

def rename_columns(df, drop_initial, drop_secondary, drop_tertiary):
    # copy of df
    df_rename = df.copy()
    
    # take away estimate
    df_rename.columns = df_rename.columns.str.replace('Estimate!!', '')
    
    # rename GRAPI - initial
    df_rename.columns = df_rename.columns.str.replace('GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI)', 'GRAPI', regex=False)
    
    # rename SMOC - initial
    df_rename.columns = df_rename.columns.str.replace('SELECTED MONTHLY OWNER COSTS (SMOC)', 'SMOC', regex=False)
    
    # rename SMOCAPI - initial
    df_rename.columns = df_rename.columns.str.replace('SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)', 'SMOCAPI', regex=False)
    
    # rename GRAPI - secondary
    df_rename.columns = df_rename.columns.str.replace('Occupied units paying rent (excluding units where GRAPI cannot be computed)', 'Occupied units paying rent', regex=False)
    
    # rename SMOCAPI - secondary with mortgage information
    df_rename.columns = df_rename.columns.str.replace('Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)', 'Housing units with a mortgage', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)', 'Housing units without a mortgage', regex=False)
    
    # rename second tiers
    df_rename.columns = df_rename.columns.str.replace('Total housing units', 'Total', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Occupied units paying rent', 'Rent', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Occupied housing units', 'Occupied', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Vacant housing units', 'Vacant', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Owner-occupied units', 'Owner Occupied', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Housing units with a mortgage', 'With Mortgage', regex=False)
    df_rename.columns = df_rename.columns.str.replace('Housing units without a mortgage', 'Without Mortgage', regex=False)
    
    # non-static columns
    static_columns = ['Year', 'Geography', 'Geographic Area Name']
    df_cols = [col for col in df_rename.columns if col not in static_columns]
    
    # apply tier_columns function
    df_tiers = pd.DataFrame(tier_columns(df_cols))
    df_tiers.columns = [f'tier_{col}' for col in range(df_tiers.shape[1])]
    
    # drop with null
    drop_cols = []
    for index, row in df_tiers.iterrows():
        tier_0 = row['tier_0']
        tier_1 = row['tier_1']
        tier_2 = row['tier_2']
        if (tier_2 is None) & (tier_0 in drop_initial):
            drop_cols.append(f'{tier_0}!!{tier_1}')
        elif(tier_2 is not None) & (tier_0 in drop_secondary):
            drop_cols.append(f'{tier_0}!!{tier_1}!!{tier_2}')
        
    drop_cols = drop_cols + drop_tertiary
    retain_cols = [col for col in df_rename.columns if col not in drop_cols]
    retain_cols.sort()
    df_final = df_rename[retain_cols]
    
    return df_final


df_renamed = rename_columns(df, drop_initial, drop_secondary, drop_tertiary)
new_cols = df_renamed.columns
new_tiers = pd.DataFrame(tier_columns(new_cols))

'''
additional steps:
    - BEDROOMS: {tier_2}
    - GRAPI: GRAPI - {tier_2}
    - GROSS RENT: {tier_1} (Rent)
    - HOUSE HEATING FUEL: Heat - {tier_2}
    - HOUSING OCCUPANCY: {tier_2} (Occupied or Vacant)
    - HOUSING TENURE: {tier_2} (Owner-occupied or Renter-occupied)
    - MORTGAGE STATUS: {tier_2} (With Mortgage)
    - OCCUPANTS PER ROOM: Occupants - {tier_2}
    - ROOMS: {tier_2}
    - SELECTED CHARACTERISTICS: {tier_2} (shows lacking services)
    - SMOCAPI: {tier_0} - {tier_1} - {tier_2}
    - UNITS IN STRUCTURE: {tier_2}
    - VALUE: {tier_2}
    - VEHICLES AVAILABLE: {tier_2}
    - YEAR STRUCTURE BUILT: {tier_2}
'''

def finalize_restructure(df):
    df_cols = df.columns
    df_tiers = pd.DataFrame(tier_columns(df_cols))
    df_tiers.columns = [f'tier_{col}' for col in range(df_tiers.shape[1])]
    
    col_remapping = {}
    for index, row in df_tiers.iterrows():
        tier_0 = row['tier_0']
        tier_1 = row['tier_1']
        tier_2 = row['tier_2']
        
        if tier_0 == 'BEDROOMS':
            # n bedroom(s)
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'GRAPI':
            # GRAPI - percent
            col_remapping[df_cols[index]] = f'GRAPI - {tier_2}'
        elif tier_0 == 'GROSS RENT':
            # Rent
            col_remapping[df_cols[index]] = tier_1
        elif tier_0 == 'HOUSE HEATING FUEL':
            # Heat - type
            col_remapping[df_cols[index]] = f'Heat - {tier_2}'
        elif tier_0 == 'HOUSING OCCUPANCY':
            # either Vacant or Occupied
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'HOUSING TENURE':
            # either (Owner-occupied or Renter-occupied)
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'MORTGAGE STATUS':
            # either With or Without (both classes are Owner Occupied)
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'OCCUPANTS PER ROOM':
            # Occupants - avg
            col_remapping[df_cols[index]] = f'Occupants - {tier_2}'
        elif tier_0 == 'ROOMS':
            # n room(s)
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'SELECTED CHARACTERISTICS':
            # type of services lacking
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'SMOCAPI':
            # SMOCAPI - mortgage status - percent
            col_remapping[df_cols[index]] = f'{tier_0} - {tier_1} - {tier_2}'
        elif tier_0 == 'UNITS IN STRUCTURE':
            # type and/or number of units in structure
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'VALUE':
            # value ranges
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'VEHICLES AVAILABLE':
            # vehicle spaces available
            col_remapping[df_cols[index]] = tier_2
        elif tier_0 == 'YEAR STRUCTURE BUILT':
            # date range of structures built
            col_remapping[df_cols[index]] = tier_2
            
    # apply remapping
    df_final = df.rename(columns=col_remapping)
    
    return df_final

df_final = finalize_restructure(df_renamed)

# fix N in numeric columns and make type
def fix_numeric_cols(df):
    ignore_cols = ['Year', 'Geography', 'Geographic Area Name']
    fix_cols = [col for col in df.columns if col not in ignore_cols]
    df.loc[:, fix_cols] = df.loc[:, fix_cols].replace('N', 0)
    df.loc[:, fix_cols] = df.loc[:, fix_cols].replace('(X)', 0)
    
    issue_cols = []
    for col in fix_cols:
        try:
            df[col] = df[col].astype(int)
        except ValueError:
            issue_cols.append(col)
            
    return df, issue_cols

final_df, issues = fix_numeric_cols(df_final)

final_describe = final_df.describe()

final_df.to_csv('../data/cleaned_df.csv', index=False)

'''
# rows with null values in third tier?
df_tiers_null = df_tiers[df_tiers.isnull().any(axis=1)]
df_tiers_not_null = df_tiers[df_tiers.notnull().all(axis=1)]

for _, row in df_tiers_null.iterrows():
    print(f'{row["tier_1"]} - {row["tier_2"]}')



Initial Results:

BEDROOMS - Total housing units
GROSS RENT - Occupied units paying rent
GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME (GRAPI) - Occupied units paying rent (excluding units where GRAPI cannot be computed)
HOUSE HEATING FUEL - Occupied housing units
HOUSING OCCUPANCY - Total housing units
HOUSING TENURE - Occupied housing units
MORTGAGE STATUS - Owner-occupied units
OCCUPANTS PER ROOM - Occupied housing units
ROOMS - Total housing units
SELECTED CHARACTERISTICS - Occupied housing units
SELECTED MONTHLY OWNER COSTS (SMOC) - Housing units with a mortgage
SELECTED MONTHLY OWNER COSTS (SMOC) - Housing units without a mortgage
SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI) - Housing units with a mortgage (excluding units where SMOCAPI cannot be computed)
SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI) - Housing unit without a mortgage (excluding units where SMOCAPI cannot be computed)
UNITS IN STRUCTURE - Total housing units
VALUE - Owner-occupied units
VEHICLES AVAILABLE - Occupied housing units
YEAR HOUSEHOLDER MOVED INTO UNIT - Occupied housing units
YEAR STRUCTURE BUILT - Total housing units

drop_with_null = ['BEDROOMS', 'GRAPI', 'HOUSE HEATING FUEL', 'HOUSING OCCUPANCY', 'HOUSING TENURE', 'MORTGAGE STATUS', 'OCCUPANTS PER ROOM', 'ROOMS', 'SMOC', 'SMOPCAPI', 'UNITS IN STRUCTURE', 'VALUE', 'VEHICLES AVAILABLE', 'YEAR HOUSEHOLDER MOVED INTO UNIT', 'YEAR STRUCTURE BUILT']
drop_with_no_null = ['GROSS RENT', 'ROOMS - MEDIAN', 'SMOC', 'VALUE - MEDIAN', 'YEAR HOUSEHOLDER MOVED INTO UNIT']

Thoughts:
    - bedrooms: does this represent a studio? if it does not, seems to have no value added
        - there is a no bedroom option, other options are 1, 2, 3, 4 and 5+
        - vote: drop column
    - gross rent: total units paying rent could be better indicator of the inadequate categories:
        - categories are 1k-1.5k and median
        - vote: retain column, drop others
    - grapi:
        - categories are adequate
        - vote: drop column
    - house heating fuel:
        - there is an "other" option, with adequate categories otherwise
        - vote: drop column
    - housing occupancy:
        - categories are vacant or occupied
        - vote: drop column
    - housing tenure:
        - categories are owner-occupied or renter-occupied
        - vote: drop column
    - mortgage status:
        - categories are with and without a mortgage
        - vote: drop column
    - occupants per room:
        - categories are 1 or less, 1.01 to 1.5, and 1.51 +
        - vote: drop column
    - rooms: is this total rooms (i.e. bedroom + bathroom) or does it represent apartment type buildings?
        - categories include 1, ..., 8, 9+, and median
        - vote: unsure
    - selected characteristics: looks like defining services which are not available at residences
        - categories: no telephone, lacking complete kitchen, and lacking complete plumbing
        - is the null third tier a total of these, or do they represent other lack of services?
        - vote: compare totals -> if total is a total of selected characteristics -> drop -> else -> unsure
    - SMOC - Mortgage and SMOC - No Mortgage:
        - categories aren't adequate and a total doesn't make sense
        - drop ALL SMOC columns
    - SMOCAPI:
        - with a mortgage (5) categories: <20, 20-24.9, 25-29.9, 30-34.9, 35+
        - without a mortgage (7) categories: <10, 10-14.9, 15-19.9, 20-24.9, 25-25.9, 30-.34.9, 35+
        - df_tiers_not_null[df_tiers_not_null['tier_1']=='SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME (SMOCAPI)'].value_counts()
        - vote: drop columns of SMOCAPI - with mortgage and SMOCAPI - no mortgage (retain the categorical interpretations)
    - units in structure: is this additional storage or like garage/trailer park/apartment ownwers?
        - vote: unsure... drop all or drop total
    - value:
        - vote: drop total columns
        - vote: drop median column? could minconstrue things  
    - vehicles available:
        - vote: drop column
    - year householder moved into unit:
        - vote: drop ALL columns
    - year structure built:
        - primary vote: do we want to get housing units built updated for each 5 year period?
        - secondary vote: drop total column, but...
        - tertiary vote: how to retain year structure built?
            - 1939 (-)
            - 1940 - 1949
            - 1950 - 1959
            - 1960 - 1969
            - 1970 - 1979
            - 1980 - 1989
            - 1990 - 1999
            - 2000 - 2009
        - since these columns are retained throughout these datasets... how do they change... and what do new structures bring us?
        - also... we can't exactly trace these individually
'''

'''
Questions:
    1. How amenities change over time (heating, lacking services, car spaces, vacancy vs occupancy) - Julia
    2. Housing structures (number of rooms) - Jess
    3. Prices (loop in percentage columns) - Carl
'''

# analysis columns
dollar_columns = [col for col in df_final.columns if col.startswith('$')]
grapi_columns = [col for col in df_final.columns if col.startswith('GRAPI')]
smocapi_columns = [col for col in df_final.columns if col.startswith('SMOCAPI')]
info_columns = ['Year', 'Geography', 'Geographic Area Name']
analysis_cols = info_columns + dollar_columns + grapi_columns + smocapi_columns

# analysis dataframe
analysis_df = final_df[analysis_cols]

# analysis county subset
county_df = analysis_df[analysis_df['Geography'].str.startswith('05')]
county_df.reset_index(drop=True, inplace=True)

# state and county columns
county_df[['County', 'State']] = county_df['Geographic Area Name'].apply(lambda row: pd.Series(row.split(', ')))

# make year int
county_df['Year'] = county_df['Year'].astype(int)

'''
- normalized results in each county
- top n for category in each state
    - i.e. county with most 1m dollar homes vs county with most 50k homes
- top n across the country
'''

# get coordinates for each county
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent='housing_analysis')
county_info = county_df.iloc[0]['Geographic Area Name']
location = geolocator.geocode(county_info)
location.latitude
location.longitude

