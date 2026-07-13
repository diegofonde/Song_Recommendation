# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:00:22 2026

@author: dbf98
"""
# Imports
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

## Importing preprocessed listening history
listening_history = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\user_history.pkl')
listening_history.head()

# Getting the list of unique users from the dataset 
unique_ids = listening_history['user_id'].unique() # Important so that in the test set we won't be predicting the tags of users that has data already inputted in the training set

cv_training_set_ids, test_set_ids = train_test_split(
    unique_ids,
    test_size = 0.2,
    random_state = 42
)

test_set_ids, validation_set_ids = train_test_split(
    test_set_ids,
    test_size = 0.5,
    random_state = 42
)