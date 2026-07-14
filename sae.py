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

max_test = test_set['track_id'].max()
max_validation = validation_set['track_id'].max()
max_train = cv_training_set['track_id'].max()

num_tracks = max(max_test, max_validation, max_train)
print(num_tracks)

# Properly convertining data into dictionary format, ready for SAE input
def convert(data):
    new_data = {}
    for user_id, track_id, log_playcount in zip(data['user_id'], data['track_id'], data['log_playcount']): # For loop loops through every column not row for pd
        
        if user_id not in new_data:
            new_data[user_id] = {}
            
        new_data[user_id][track_id] = log_playcount
        
    return new_data

test_set = convert(test_set)
validation_set = convert(validation_set)
cv_training_set = convert(cv_training_set)  


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
        
    for id_user in cv_training_set.keys():
        
        user_vector = torch.zeros(num_tracks, dtype = torch.float32) # Makes initial vector of 0s that represent the playcount a user has for a specific track_id
        user_history = cv_training_set[id_user] # Gets the actual listening history of the user
        
        for track_id, log_playcount in user_history.items():
            user_vector[track_id - 1] = log_playcount
            
        inputs = user_vector.unsqueeze(0) # Formats data in to [1, num_tracks]
        target = inputs.clone() # Actually values to be compared post decoder
        
        if torch.sum(target.data > 0) > 0:
            
            output = sae.forward(inputs)
            output = torch.where(target == 0, 0.0, output) # Makes it so that log playcount values that are actually 0 will automatically be predicted as 0, helps with model performance
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            s += 1.
            optimizer.step()
            
    print('epoch: ' + str(epoch) + 'Loss: ' + str(train_loss/s))
            
        
        
    
        
        
        
        