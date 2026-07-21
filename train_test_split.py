# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:00:22 2026

@author: dbf98
"""
# Imports

## Data Manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

## Data Exportation
import os

# Importing preprocessed listening history
listening_history = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\user_history.pkl')
listening_history.head()

# Cutting down dataset to only contain the tracks with 5 or more total log_playcounts
listening_history_tracks = listening_history.groupby(by = 'track_id')['log_playcount'].sum().reset_index()
listening_history_tracks.head()
listening_history_tracks.describe()
listening_history_tracks_top_playcount = listening_history_tracks[listening_history_tracks['log_playcount'] >= 5]
listening_history_tracks_top_playcount.describe()
top_playcount_track_id =  listening_history_tracks_top_playcount['track_id'].unique()

# Filtering out
listening_history_filtered = listening_history[listening_history['track_id'].isin(top_playcount_track_id)]

# Getting the list of unique users from the dataset 
unique_ids = listening_history_filtered['user_id'].unique() # Important so that in the test set we won't be predicting the tags of users that has data already inputted in the training set

cv_training_set_ids, test_set_ids = train_test_split(
    unique_ids,
    test_size = 0.4,
    random_state = 42
)

test_set_ids, validation_set_ids = train_test_split(
    test_set_ids,
    test_size = 0.5,
    random_state = 42
)

test_set = listening_history_filtered[listening_history_filtered['user_id'].isin(test_set_ids)]
validation_set = listening_history_filtered[listening_history_filtered['user_id'].isin(validation_set_ids)]
cv_training_set = listening_history_filtered[listening_history_filtered['user_id'].isin(cv_training_set_ids)]

# Visualizing test set, validation set and training set to ensure that the distribution of playcount data is consistent throughout

## Visualizing testing_set playcounts
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(test_set['log_playcount'], bins = 100, kde = True, ax = ax, color = 'teal')
ax.set_title('Test set Log-Playcount distribution')
ax.set_xlabel('Test set Log-Playcount')
ax.set_ylabel('Number of songs')
plt.tight_layout()
plt.show()

## Visualizing validation_set playcounts
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(validation_set['log_playcount'], bins = 100, kde = True, ax = ax, color = 'teal')
ax.set_title('Validation set Log-Playcount distribution')
ax.set_xlabel('Validation set Log-Playcount')
ax.set_ylabel('Number of songs')
plt.tight_layout()
plt.show()

## Visualizing training_set playcounts
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(cv_training_set['log_playcount'], bins = 100, kde = True, ax = ax, color = 'teal')
ax.set_title('Training set Log-Playcount distribution')
ax.set_xlabel('Training set Log-Playcount')
ax.set_ylabel('Number of songs')
plt.tight_layout()
plt.show()

## Printing out log_playcount descriptions to double check 
print(test_set['log_playcount'].describe())
print(validation_set['log_playcount'].describe())
print(cv_training_set['log_playcount'].describe())

# Exporting datasets since the distributions are acceptable
test_set.to_parquet(os.path.join(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits', 'test_set.pkl'))
validation_set.to_parquet(os.path.join(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits', 'validation_set.pkl'))
cv_training_set.to_parquet(os.path.join(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits', 'cv_training_set.pkl'))