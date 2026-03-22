import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data):
    """Scale data using MinMaxScaler (0-1 normalization)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)