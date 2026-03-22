import matplotlib.pyplot as plt
import os

def plot_predictions(actual, lstm_preds, gru_preds, show=False):
    # Create output directory if it doesn't exist
    os.makedirs("outputs/plots", exist_ok=True)
    
    plt.figure(figsize=(12,6))
    
    plt.plot(actual[:,0], label="Actual")
    plt.plot(lstm_preds[:,0], label="LSTM")
    plt.plot(gru_preds[:,0], label="GRU")
    
    plt.legend()
    plt.title("Model Comparison")
    plt.savefig("outputs/plots/prediction.png")
    
    if show:
        plt.show()
    else:
        plt.close()