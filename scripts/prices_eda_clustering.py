'''
A Finalized Setup for the Prices Analysis
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# county coordinates dataset (obtained through obtain_coordinates() function)
county_coordinates_df = pd.read_csv('../data/county_coordinates.csv')

# merge coordinates into county specific dataset
df = pd.merge(county_df, county_coordinates_df, on='Geographic Area Name')

# normalize the pricing families across the dataset (i.e. counts are turned into percentages according to county and family type)
# step 1: get count totals across the counties and analysis families
df['Price Total'] = df[price_columns].sum(axis=1)
df['GRAPI Total'] = df[grapi_columns].sum(axis=1)
smocapi_with_columns = [col for col in df.columns if col.startswith('SMOCAPI - With Mortgage')]
smocapi_without_columns = [col for col in df.columns if col.startswith('SMOCAPI - Without Mortgage')]
df['SMOCAPI With Mortgage Total'] = df[smocapi_with_columns].sum(axis=1)
df['SMOCAPI Without Mortgage Total'] = df[smocapi_without_columns].sum(axis=1)

# step 2: normalize via percents across the analysis families
df_percent = df.copy()
df_percent[price_columns] = df_percent[price_columns].div(df_percent['Price Total'], axis=0)
df_percent[grapi_columns] = df_percent[grapi_columns].div(df_percent['GRAPI Total'], axis=0)
df_percent[smocapi_with_columns] = df_percent[smocapi_with_columns].div(df_percent['SMOCAPI With Mortgage Total'], axis=0)
df_percent[smocapi_without_columns] = df_percent[smocapi_without_columns].div(df_percent['SMOCAPI Without Mortgage Total'], axis=0)

# function to cluster the normalized dataset by an analysis family
def price_clustering(df_percent, price_columns, grapi_columns, smocapi_columns, info_columns, subset):
    # pick pricing subset
    if subset == 'All':
        pca_numericals = price_columns + grapi_columns + smocapi_columns
    elif subset == 'Prices':
        pca_numericals = price_columns
    elif subset == 'GRAPI':
        pca_numericals = grapi_columns
    elif subset == 'SMOCAPI':
        pca_numericals = smocapi_columns
    
    # create pca - take 90% of explained variance
    # normalize
    scaler = StandardScaler()
    normalized_numericals = scaler.fit_transform(df_percent[pca_numericals])
    normalized_df = pd.DataFrame(normalized_numericals, columns=pca_numericals)
    normalized_df.fillna(0, inplace=True)
    
    # run pca
    pca = PCA()
    pca.fit(normalized_df)

    # pca projection space
    pca_projection = pca.transform(normalized_df)
    pca_df = pd.DataFrame(pca_projection)
    
    # find 90% retention
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    retained_90 = np.argmax(cumulative_variance > 0.9) + 1
    print(f'Columns Required: {retained_90 / len(cumulative_variance):.2%}')
    
    # creating the dataframe
    cluster_df = df_percent[info_columns + ['Latitude', 'Longitude']].copy()
    
    # kmeans clustering
    for n_cluster in range(2, 7):
        model = KMeans(n_clusters=n_cluster, random_state=42)
        model.fit(pca_df.iloc[:, :retained_90])
        model_labels = model.predict(pca_df.iloc[:, :retained_90])
        cluster_df[f'Cluster_{n_cluster}'] = model_labels
    
    return cluster_df

# run clustering across the families
cluster_df_all = price_clustering(df_percent, price_columns, grapi_columns, smocapi_columns, info_columns, subset='All')
cluster_df_prices = price_clustering(df_percent, price_columns, grapi_columns, smocapi_columns, info_columns, subset='Prices')
cluster_df_grapi = price_clustering(df_percent, price_columns, grapi_columns, smocapi_columns, info_columns, subset='GRAPI')
cluster_df_smocapi = price_clustering(df_percent, price_columns, grapi_columns, smocapi_columns, info_columns, subset='SMOCAPI')

# illustrate the clustering results
def visualize_clusters(cluster_df, family_name, custom_palette=sns.color_palette()):
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f'Clustering Across {family_name} Columns', y=1.02, fontsize=16)
    clusters = range(2, 7)
    years = [2013, 2018, 2023]
    for cluster_count, n_cluster in enumerate(clusters):
        for year_count, year in enumerate(years):
            sns.scatterplot(cluster_df[cluster_df['Year']==year], x='Longitude', y='Latitude', hue=f'Cluster_{n_cluster}', ax=axes[cluster_count, year_count], palette=custom_palette[:n_cluster])
            axes[cluster_count, year_count].set_title(f'Year: {year}, Clusters: {n_cluster}')
            axes[cluster_count, year_count].legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'images/price_ranges_clustering_{family_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# visualize clustering for the price family
visualize_clusters(cluster_df_prices, 'Price')

# visualize clustering for the grapi family
visualize_clusters(cluster_df_grapi, 'GRAPI')

# visualize clustering for the smocapi family
visualize_clusters(cluster_df_smocapi, 'SMOCAPI')

'''
The clustering function uses 90% explained variance from PCA, the PCA columns required are:
    - All Families: 61.54%
    - Price Family: 50.00%
    - GRAPI Family: 83.33%
    - SMOCAPI Family: 66.67%
'''

# function to get metrics for clusters (i.e. is there something specific that made these clusters?)
def cluster_metrics(cluster_df, df_percent, interest_columns):
    # subset by year
    years = cluster_df['Year'].unique().tolist()
    cluster_cols = [col for col in cluster_df.columns if col.startswith('Cluster')]
    
    # metrics storage
    metrics = {'Year': [], 'Cluster': [], 'Mean': [], 'Std': [], 'Median': [], 'Variable': [], 'Total Clusters': []}
    
    # subset by cluster
    for cluster in cluster_cols:
        # get the uniques clusters within each cluster specification
        clusters_within = cluster_df[cluster].unique().tolist()
        # iterate on specific clusters
        for spec_cluster in clusters_within:
            # iterate on years
            for year in years:
                # get counties for specific subset
                cluster_counties = cluster_df[(cluster_df['Year'] == year) & (cluster_df[cluster] == spec_cluster)]['Geography']
                # get df_percent specific subset
                subset = df_percent[(df_percent['Year'] == year) & (df_percent['Geography'].isin(cluster_counties))][interest_columns]
                # calculate means on the df_percent columns of interest
                means = subset.mean()
                # calculate medians on the df_percent columns of interest
                medians = subset.median()
                # calculate standard deviations on the df_percent columns of interest
                stds = subset.std()
                # extract variable specific metrics
                for variable in interest_columns:
                    metrics['Year'].append(year)
                    metrics['Cluster'].append(spec_cluster)
                    metrics['Variable'].append(variable)
                    metrics['Mean'].append(means[variable])
                    metrics['Median'].append(medians[variable])
                    metrics['Std'].append(stds[variable])
                    metrics['Total Clusters'].append(max(clusters_within) + 1)
                    
    # return dataframe with results
    return pd.DataFrame(metrics)

# explore cluster metrics across the families
metrics_price_cluster = cluster_metrics(cluster_df_prices, df_percent, price_columns)
metrics_grapi_cluster = cluster_metrics(cluster_df_grapi, df_percent, grapi_columns)
metrics_smocapi_cluster = cluster_metrics(cluster_df_smocapi, df_percent, smocapi_columns)

# color palette
custom_palette = sns.color_palette()

# ordering for analysis families - price
price_order = ['$1,000,000 or more', '$500,000 to $999,999', '$300,000 to $499,999', '$200,000 to $299,999', '$150,000 to $199,999', '$100,000 to $149,999', '$50,000 to $99,999', 'Less than $50,000']
price_labels = ['\$1M +', '\$500k - \$1M', '\$300k - \$500k', '\$200k - \$300k', '\$150k - \$200k', '\$100k - \$150k', '\$50k - \$100k', 'Less Than $50k']

# ordering for analysis families - grapi
grapi_order = ['GRAPI - 35.0 percent or more', 'GRAPI - 30.0 to 34.9 percent', 'GRAPI - 25.0 to 29.9 percent', 'GRAPI - 20.0 to 24.9 percent', 'GRAPI - 15.0 to 19.9 percent', 'GRAPI - Less than 15.0 percent']
grapi_labels = ['35% +', '30% - 35%', '25% - 30%', '20% - 25%', '15% - 20%', 'Less Than 15%']

# ordering for analysis families - smocapi with mortgage
smocapi_with_order = ['SMOCAPI - With Mortgage - 35.0 percent or more', 'SMOCAPI - With Mortgage - 30.0 to 34.9 percent', 'SMOCAPI - With Mortgage - 25.0 to 29.9 percent', 'SMOCAPI - With Mortgage - 20.0 to 24.9 percent', 'SMOCAPI - With Mortgage - Less than 20.0 percent']
smocapi_with_labels = ['35% +', '30% - 35%', '25% - 30%', '20% - 25%', 'Less Than 20%']

# ordering for analysis families - smocapi without mortgage
smocapi_without_order = ['SMOCAPI - Without Mortgage - 35.0 percent or more', 'SMOCAPI - Without Mortgage - 30.0 to 34.9 percent', 'SMOCAPI - Without Mortgage - 25.0 to 29.9 percent', 'SMOCAPI - Without Mortgage - 20.0 to 24.9 percent', 'SMOCAPI - Without Mortgage - 15.0 to 19.9 percent', 'SMOCAPI - Without Mortgage - 10.0 to 14.9 percent', 'SMOCAPI - Without Mortgage - Less than 10.0 percent']
smocapi_without_labels = ['35% +', '30% - 35%', '25% - 30%', '20% - 25%', '15% - 20%', '10% - 15%', 'Less Than 10%']

# ordering for analysis families - smocapi overall
smocapi_order = smocapi_with_order + smocapi_without_order
smocapi_labels = [f'(Mortgage) {label}' for label in smocapi_with_labels] + [f'(No Mortgage) {label}' for label in smocapi_without_labels]

# function to visualize the cluster metrics
def visualize_cluster_metrics(metrics_df, family_order, family_labels, family_name, metrics_type='Mean', custom_pallete=sns.color_palette()):
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f'Cluster Metrics: {family_name}', y=1.02, fontsize=16)
    clusters = range(2, 7)
    years = [2013, 2018, 2023]
    for cluster_count, n_cluster in enumerate(clusters):
        for year_count, year in enumerate(years):
            metrics_subset = metrics_df[(metrics_df['Year']==year) & (metrics_df['Total Clusters']==n_cluster)]
            sns.barplot(metrics_subset, x=metrics_type, y='Variable', hue='Cluster', order=family_order, ax=axes[cluster_count, year_count], palette=custom_palette[:n_cluster])
            axes[cluster_count, year_count].set_yticks(np.arange(len(family_labels)))
            axes[cluster_count, year_count].set_yticklabels(family_labels)
            axes[cluster_count, year_count].set_title(f'Year: {year}, Clusters: {n_cluster}')
            axes[cluster_count, year_count].legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'images/cluster_metrics_{family_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

# visualize cluster metrics for the price family
visualize_cluster_metrics(metrics_price_cluster, price_order, price_labels, 'Price', metrics_type='Mean')

# visualize cluster metrics for the grapi family
visualize_cluster_metrics(metrics_grapi_cluster, grapi_order, grapi_labels, 'GRAPI', metrics_type='Mean')

# visualize cluster metrics for the smocapi family
visualize_cluster_metrics(metrics_smocapi_cluster, smocapi_order, smocapi_labels, 'SMOCAPI', metrics_type='Mean')

'''
2 and 3 clusters seem to give a decent representation of how the different money related families
cluster into regions across the US.

Simplify the visualizations to show the distributions with the clusters for 2 and 3 clustering amounts.
'''

# function to produce simplified visualizations for 2 and 3 clusters
def simplify_visualizations(cluster_df, metrics_df, family_order, family_labels, family_name, metrics_type='Mean', custom_palette=sns.color_palette()):
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle(f'Cluster Diagnostics: {family_name}', y=1.02, fontsize=16)
    clusters = range(2, 4)
    years = [2013, 2018, 2023]
    
    # starting row
    row_count = 0
    
    for n_cluster in clusters:
        for year_count, year in enumerate(years):
            # cluster label percentages
            cluster_percents = cluster_df[cluster_df['Year']==year][f'Cluster_{n_cluster}'].value_counts(normalize=True) * 100
            cluster_percents = cluster_percents.round(1)
            
            # cluster visualizations
            sns.scatterplot(cluster_df[cluster_df['Year']==year], x='Longitude', y='Latitude', hue=f'Cluster_{n_cluster}', ax=axes[row_count, year_count], palette=custom_palette[:n_cluster])
            axes[row_count, year_count].set_title(f'Year: {year}, Clusters: {n_cluster}')
            # legend customization
            handles, labels = axes[row_count, year_count].get_legend_handles_labels()
            new_labels = [f'Cluster {label}: {cluster_percents[int(label)]}%' for label in labels]
            axes[row_count, year_count].legend(handles=handles, labels=new_labels, loc='best')
            
            # metrics visualizations
            metrics_subset = metrics_df[(metrics_df['Year']==year) & (metrics_df['Total Clusters']==n_cluster)]
            sns.barplot(metrics_subset, x=metrics_type, y='Variable', hue='Cluster', order=family_order, ax=axes[row_count + 1, year_count], palette=custom_palette[:n_cluster])
            axes[row_count + 1, year_count].set_yticks(np.arange(len(family_labels)))
            axes[row_count + 1, year_count].set_yticklabels(family_labels)
            axes[row_count + 1, year_count].set_title(f'Year: {year}, Clusters: {n_cluster}')
            axes[row_count + 1, year_count].legend(loc='best')
            
        # increase row count with cluster iteration
        row_count += 2
        
    plt.tight_layout()
    plt.savefig(f'images/simplified_{family_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

# visualize simplified clusters results for the price family
simplify_visualizations(cluster_df_prices, metrics_price_cluster, price_order, price_labels, 'Price')

# visualize simplified clusters results for the grapi family
simplify_visualizations(cluster_df_grapi, metrics_grapi_cluster, grapi_order, grapi_labels, 'Grapi')

# visualize simplified clusters results for the smocapi family
simplify_visualizations(cluster_df_smocapi, metrics_smocapi_cluster, smocapi_order, smocapi_labels, 'SMOCAPI')
