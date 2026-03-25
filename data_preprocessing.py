# Imports

## Data Manipulation
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

music_info = pd.read_csv('data/raw/Music Info.csv')
music_info.head()

listening_history = pd.read_csv('data/raw/User Listening History.csv')
listening_history.head()

music_info['genre'].head()
music_info['tags'].head()

music_info.columns
listening_history.columns

## Visualizing genre + tag distribution
print(music_info['tags'].isna().sum()) # Checking to see how many NA rows are there for tags 
print(music_info['genre'].isna().sum()) # Checking to see how many NA row are there for genre 
total_na = music_info['tags'].isna() & music_info['genre'].isna()
print(total_na.sum())

## Genre
genre_counts = music_info['genre'].value_counts()
genre_counts.head()
plt.figure(figsize=(12,8))
sns.barplot(x = genre_counts.values, y = genre_counts.index, palette = 'viridis')
plt.title('Genre distribution', fontsize = 15)
plt.xlabel('Genre Count', fontsize = 12)
plt.ylabel('Genre', fontsize = 12)

## Tags
tags_list = music_info['tags'].dropna().str.split(',') # Separate tags by comma
flatten_tags_list = tags_list.explode().str.strip()
tag_count = flatten_tags_list.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(x = tag_count.values, y = tag_count.index, palette = 'viridis')
plt.title('Tag distribution', fontsize = 15)
plt.xlabel('Tag Count', fontsize = 12)
plt.ylabel('Tag', fontsize = 12)
print(len(tag_count.index))

## Visualizing year
print(music_info['year'].describe())

music_info['era'] = music_info['year'].astype(str)
music_info.loc[music_info['year'] < 1950, 'era'] = "Pre-1950s"
music_info.loc[(music_info['year'] >= 1950) & (music_info['year'] < 1960), 'era'] = "1950s"
music_info.loc[(music_info['year'] >= 1960) & (music_info['year'] < 1970), 'era'] = "1960s"
music_info.loc[(music_info['year'] >= 1970) & (music_info['year'] < 1980), 'era'] = "1970s"
music_info.loc[(music_info['year'] >= 1980) & (music_info['year'] < 1990), 'era'] = "1980s"
music_info.loc[(music_info['year'] >= 1990) & (music_info['year'] < 2000), 'era'] = "1990s"
music_info.loc[(music_info['year'] >= 2000) & (music_info['year'] < 2010), 'era'] = "2000s"
music_info.loc[(music_info['year'] >= 2010) & (music_info['year'] < 2020), 'era'] = "2010s"
music_info.loc[music_info['year'] >= 2020, 'era'] = "2020s"

era_distribution = music_info['era'].value_counts()
era_order = ['Pre-1950s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
plt.figure(figsize =(12,5))
ax = sns.barplot(x = era_distribution.index, y = era_distribution.values, order = era_order, palette = 'viridis')
ax.bar_label(ax.containers[0], padding = 3, fontsize = 10)
plt.title('Era distribution', fontsize = 15)
plt.xlabel('Era', fontsize = 12)
plt.ylabel('Era count', fontsize = 12)
plt.ylim(0, era_distribution.max() * 1.1)
plt.show()

# year_distribution = music_info['year'].value_counts().sort_index()
# plt.figure(figsize=(12,8))
# sns.barplot(x = year_distribution.index, y = year_distribution.values, palette = 'viridis')
# plt.title('Year distribution', fontsize = 15)
# plt.xlabel('Year', fontsize = 12)
# plt.ylabl('Year count', fontsize = 12)
# print(year_distribution.index)


## Visualizing continious music data
continuous = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
music_info[continuous].hist(bins=30, figsize=(12, 8), color='teal')
plt.suptitle("Continuous Feature Distributions")
plt.show()

## Visualizing categorical music data
categorical = ['key', 'mode', 'time_signature']
total = len(music_info)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(categorical):
    sns.countplot(x=music_info[col], ax=axes[i], palette='viridis')
    axes[i].set_title(f'Frequency of {col}')
    
    for p in axes[i].patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        axes[i].annotate(percentage, (x, y), 
                         ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

## Visualizing the playcounts
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(np.log1p(listening_history['playcount']), bins = 100, kde = True, ax = ax, color = 'teal')
ax.set_title = ('Log-Playcount distribution')
ax.set_xlabel = ('Log-Playcount')
ax.set_ylabel = ('Number of songs')
plt.tight_layout()
plt.show()

# Checking the biggest values for playcount in preparation for data preprocessing
print(listening_history['playcount'].max())
print(listening_history['playcount'].nlargest(50))

# Data Preprocessing
music_info_export = music_info
listening_history_export = listening_history

## Quantile transforming extremely skewed data 
quantile_values = ['acousticness', 'instrumentalness', 'liveness']
qt = QuantileTransformer(output_distribution = 'normal', n_quantiles = 1000, random_state = 42)
music_info_export[quantile_values] = qt.fit_transform(music_info[quantile_values])

## MinMax scaling the rest of the quantitative variables
quantitative_values = ['danceability', 'energy', 'loudness', 'speechiness', 'valence', 'tempo', 'duration_ms']
sc = MinMaxScaler()
music_info_export[quantitative_values] = sc.fit_transform(music_info[quantitative_values])



