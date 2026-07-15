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

max_test_track = test_set['track_id'].max()
max_validation_track = validation_set['track_id'].max()
max_train_track = cv_training_set['track_id'].max()
num_tracks = max(max_test_track, max_validation_track, max_train_track)
print(num_tracks)

max_test_user = test_set['user_id'].max()
max_validation_user = validation_set['user_id'].max()
max_train_user = cv_training_set['user_id'].max()
num_users = max(max_test_user, max_validation_user, max_train_user)
print(num_users)


# Properly convertining data into dictionary format, ready for SAE input
def convert(data):
    
    new_data = {}
    
    for user_id, group in data.groupby('user_id'):
        
        # Create a single, blank vector just for this specific user
        user_vector = torch.zeros(num_tracks, dtype = torch.float32)
        
        # Pull out the tracks and playcounts for this user
        track_ids = group['track_id'].values - 1
        playcounts = group['log_playcount'].values
        
        # Fill just this user's vector
        user_vector[track_ids] = torch.tensor(playcounts, dtype=torch.float32)
        
        # Save it to our dictionary
        new_data[user_id] = user_vector
        
    return new_data

test_set_converted = convert(test_set)
del test_set
validation_set_converted = convert(validation_set)
del validation_set
cv_training_set_converted = convert(cv_training_set)  
del cv_training_set

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
    weight_decay=1e-5
)

nb_epoch = 50
# Training SAE
for epoch in range(0, nb_epoch):
    train_loss = 0
    s = 0.
        
    for id_user in cv_training_set_converted.keys():
        
        user_history = cv_training_set_converted[id_user] # Gets the vectorized listening history of the user
        
        inputs = user_history.unsqueeze(0) # Formats data in to [1, num_tracks]
        target = inputs.clone() # Actually values to be compared post decoder
        
        if torch.sum(target.data > 0) > 0:
            
            output = sae.forward(inputs)
            output = torch.where(target == 0, 0.0, output) # Makes it so that log playcount values that are actually 0 will automatically be predicted as 0, helps with model performance
            loss = criterion(output, target)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            s += 1.
            optimizer.step()
            
    print('epoch: ' + str(epoch) + 'Loss: ' + str(train_loss/s))
            
        
        
    
        
        
        
        