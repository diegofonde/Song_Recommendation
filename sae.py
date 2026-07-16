# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:39:36 2026

@author: dbf98
"""
# Imports 

## Data Manipulation
import pandas as pd
import numpy as np
import gc

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
training_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\cv_training_set.pkl')

max_test_track = test_set['track_id'].max()
max_validation_track = validation_set['track_id'].max()
max_train_track = training_set['track_id'].max()
num_tracks = max(max_test_track, max_validation_track, max_train_track)
print(num_tracks)

max_test_user = test_set['user_id'].max()
max_validation_user = validation_set['user_id'].max()
max_train_user = training_set['user_id'].max()
num_users = max(max_test_user, max_validation_user, max_train_user)
print(num_users)

test_set.head()

# Properly convertining data into sparse matrix format, ready for SAE input. This is important to optimize memory that all those 0s would usually take up if I use a dictionary/array format
def convert(data, total_tracks, total_users):
    
    # First step is getting the indices that represents all the users and all of the tracks + the log playcount values
    user_indices = data['user_id'].values - 1
    track_indices = data['track_id'].values - 1
    log_playcount_values = data['log_playcount'].values
    
    indices = torch.tensor(np.vstack([user_indices, track_indices]), dtype = torch.long) # Indices has user_indices on top of the stacked torch array representing the user listening to a track, and below it is the track that the user listen to
    values = torch.tensor(log_playcount_values, dtype = torch.float32) # Separate 1D torch array that represents the log playcounts
    
    sparse_matrix = torch.sparse_coo_tensor(indices, values, size = (num_users, num_tracks)) # Combining it all here to make a sparse matrix, should have O(1)
    
    return sparse_matrix.coalesce()
    

test_set_converted = convert(test_set, num_tracks, num_users)
del test_set
gc.collect()
validation_set_converted = convert(validation_set, num_tracks, num_users)
del validation_set
gc.collect()
training_set_converted = convert(training_set, num_tracks, num_users)  
del training_set
gc.collect()

# Creating architecture for SAE
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        # Encoder Layer
        self.fc1 = nn.Linear(num_tracks, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Decoder Layer
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, num_tracks)
        
        # Activation function 
        self.activation = nn.Sigmoid()
        
    def encoder(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x
    
    def decoder(self, x):
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    sae.parameters(), 
    lr=0.001, 
)

nb_epoch = 50
# Training SAE
for epoch in range(0, nb_epoch):
    train_loss = 0
    s = 0.
        
    for id_user in range(num_users):
        
        user_vector = training_set_converted[id_user].to_dense() # Gets the vectorized listening history of the user
        
        if torch.sum(user_vector) == 0: # If the specific split does not has any listening history for the user
            continue
        
        inputs = user_vector.unsqueeze(0) # Formats data in to [1, num_tracks]
        target = inputs.clone() # Actually values to be compared post decoder
        
        if torch.sum(target.data > 0) > 0:
            
            output = sae.forward(inputs)
            output = torch.where(target == 0, 0.0, output) # Makes it so that log playcount values that are actually 0 will automatically be predicted as 0, helps with model performance
            loss = criterion(output, target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            s += 1.
            optimizer.step()
            
    print('epoch: ' + str(epoch) + 'Loss: ' + str(train_loss/s))
            
        
        
    
        
        
        
        