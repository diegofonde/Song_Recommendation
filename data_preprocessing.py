# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

music_info = pd.read_csv('data/raw/Music Info.csv')

listening_history = pd.read_csv('data/raw/User Listening History.csv')

# Obtaining all column 
music_info.columns
listening_history.columns

# Looking at head of data 
music_info.head()
listening_history.head()