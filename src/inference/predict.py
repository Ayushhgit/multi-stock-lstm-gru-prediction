import torch
import numpy as np

def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        return model(X_test).numpy()

def predict_future(model, last_seq, steps):
    # Ensure input is a proper tensor
    if isinstance(last_seq, np.ndarray):
        last_seq = torch.tensor(last_seq, dtype=torch.float32)
    
    seq = last_seq.clone().detach()
    preds = []
    
    for _ in range(steps):
        with torch.no_grad():
            # Handle 2D input by ensuring proper shape (1, seq_len, features)
            if seq.dim() == 2:
                pred = model(seq.unsqueeze(0))
            else:
                pred = model(seq.unsqueeze(0).unsqueeze(0))
        
        # Get the prediction and reshape
        pred_val = pred.squeeze(0).detach()
        preds.append(pred_val.cpu().numpy())
        
        # Update sequence: remove first timestep and add new prediction
        if seq.dim() == 2:
            seq = torch.cat((seq[1:], pred_val.unsqueeze(0)), dim=0)
        else:
            seq = torch.cat((seq[1:], pred_val), dim=0)
    
    return np.array(preds)