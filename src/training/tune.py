from models.lstm import LSTMModel
import torch
import torch.nn as nn

def tune_hyperparameters(X_train, y_train, input_size):
    configs = [
        (32, 1, 0.001),
        (50, 2, 0.001),
        (50, 2, 0.0005)
    ]
    
    best_loss = float('inf')
    best_model = None
    
    for hidden_size, num_layers, lr in configs:
        model = LSTMModel(input_size, hidden_size, num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for _ in range(3):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model
    
    return best_model