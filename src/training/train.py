import torch
import torch.nn as nn

def train_model(model, X_train, y_train, epochs, lr, device='cpu'):
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    return model