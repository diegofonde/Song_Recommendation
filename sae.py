# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:39:36 2026

@author: dbf98
"""
# Imports 

## Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

## Modelling imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

# Metric imports
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
 
test_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\test_set.pkl')
validation_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\validation_set.pkl')
cv_training_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\cv_training_set.pkl')

test_set.head()

# Properly convertining data into dictionary format, ready for SAE input
def convert(data):
    new_data = {}
    for user_id, track_id, log_playcount in zip(data['track_id'], data['track_id'], data['log_playcount']): # For loop loops through every column not row for pd
        
        if user_id not in new_data:
            new_data[user_id] = {}
            
        new_data[user_id][track_id] = log_playcount
        
        return new_data

test_set = convert(test_set)
validation_set = convert(validation_set)
cv_training_set = convert(cv_training_set)  

# Converting this dictionary formatted data into torch tensors  
test_set = torch.FloatTensor(test_set)
validation_set = torch.FloatTensor(validation_set)
cv_training_set = torch.FloatTensor(cv_training_set)

# Creating architecture for SAE
        
        