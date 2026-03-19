# Imports

## Data Manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

music_info = pd.read_csv('data/raw/Music Info.csv')

listening_history = pd.read_csv('data/raw/User Listening History.csv')

## Obtaining all column 
music_info.columns
listening_history.columns

## Visualizing continious music data
continuous = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
music_info[continuous].hist(bins=30, figsize=(12, 8), color='teal')
plt.suptitle("Continuous Feature Distributions")
plt.show()

# Visualizing categorical music data
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

# Visualizing the playcounts
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(np.log1p(listening_history['playcount']), bins = 100, kde = True, ax = ax, color = 'teal')
ax.set_title = ('Log-Playcount distribution')
ax.set_xlabel = ('Log-Playcount')
ax.set_ylabel = ('Number of songs')
plt.tight_layout()
plt.show()

print(listening_history['playcount'].max())
