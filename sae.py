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
from sklearn.preprocessing import LabelEncoder as le

## Modelling imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

device = torch.device('cpu')
print(f"Target Device: {device}")

# Importing Datasets
test_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\test_set.pkl')
validation_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\validation_set.pkl')
training_set = pd.read_parquet(r'C:\Users\dbf98\Desktop\Python_Projects\Song_Recommendation\data\processed\splits\cv_training_set.pkl')

# Label encoding user_id and track_id
user_encoder = le()
track_encoder = le()

all_user_ids = pd.concat([test_set['user_id'], validation_set['user_id'], training_set['user_id']]).unique()
all_track_ids = pd.concat([test_set['track_id'], validation_set['track_id'], training_set['track_id']]).unique()
user_encoder.fit(all_user_ids)
track_encoder.fit(all_track_ids)

for set in [test_set, validation_set, training_set]:
    set['user_id'] = user_encoder.transform(set['user_id'])    
    set['track_id'] = track_encoder.transform(set['track_id'])    

max_test_track = test_set['track_id'].max()
max_validation_track = validation_set['track_id'].max()
max_train_track = training_set['track_id'].max()
num_tracks = max(max_test_track, max_validation_track, max_train_track) + 1
print(num_tracks)

max_test_user = test_set['user_id'].max()
max_validation_user = validation_set['user_id'].max()
max_train_user = training_set['user_id'].max()
num_users = max(max_test_user, max_validation_user, max_train_user) + 1
print(num_users)

test_set.head()

# Properly convertining data into sparse matrix format, ready for SAE input. This is important to optimize memory that all those 0s would usually take up if I use a dictionary/array format
def convert(data, total_tracks, total_users):
    
    # First step is getting the indices that represents all the users and all of the tracks + the log playcount values
    user_indices = data['user_id'].values
    track_indices = data['track_id'].values
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
    def __init__(self, tracks):
        super(SAE, self).__init__()
        # Encoder Layer
        self.fc1 = nn.Linear(tracks, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Decoder Layer
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, tracks)
        
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
        
sae = SAE(num_tracks)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    sae.parameters(), 
    lr=0.001, 
)


def get_batch(sparse_tensor, start_user, end_user):
    length = end_user - start_user
    sub_sparse = sparse_tensor.narrow(0, start_user, length)
    return sub_sparse.to_dense()
    

nb_epoch = 50
batch_size = 256

# Training SAE
for epoch in range(0, nb_epoch):
    train_loss = 0
    s = 0.
        
    for user in range(0, num_users, batch_size):
        
        batch_indices = torch.arange(user, min(user + batch_size, num_users))
        
        inputs = get_batch(training_set_converted, user, min(user + batch_size, num_users)) # Gets the vectorized listening history of the user and formats data to [250, num_tracks]
        
        if torch.sum(inputs) == 0: # If the specific split does not has any listening history for the user
            continue
        
        non_zero_mask = inputs > 0 # Actual songs that user has playcounts for
        
        if non_zero_mask.any():
            
            optimizer.zero_grad()
            output = sae(inputs)
            loss = criterion(output[non_zero_mask], inputs[non_zero_mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            s += 1.0
            
    print(f"Epoch {epoch + 1}/{nb_epoch} | Loss: {train_loss / max(s, 1.0):.6f}")
            
        
        
    
        
        
        
        