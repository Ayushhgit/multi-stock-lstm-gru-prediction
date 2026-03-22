import torch
from src.config import *
from src.data.data_loader import load_data
from src.data.preprocess import scale_data, create_sequences
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.training.train import train_model
from src.inference.predict import predict, predict_future
from src.utils.plotting import plot_predictions

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = load_data(TICKERS)

# Preprocess
scaled_data, scaler = scale_data(df.values)
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split
split = int(TRAIN_SPLIT * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

input_size = X.shape[2]

# Train LSTM
print("\nTraining LSTM Model...")
lstm = LSTMModel(input_size)
lstm = train_model(lstm, X_train, y_train, EPOCHS, LEARNING_RATE, device)

# Train GRU
print("\nTraining GRU Model...")
gru = GRUModel(input_size)
gru = train_model(gru, X_train, y_train, EPOCHS, LEARNING_RATE, device)

# Move models to CPU for inference if on GPU
lstm = lstm.to('cpu')
gru = gru.to('cpu')
X_test = X_test.to('cpu')
y_test = y_test.to('cpu')

# Predictions
print("\nMaking predictions...")
lstm_preds = predict(lstm, X_test)
gru_preds = predict(gru, X_test)

# Inverse scale - ensure 2D shape for scaler.inverse_transform
lstm_preds = scaler.inverse_transform(lstm_preds.reshape(-1, scaled_data.shape[1]))
gru_preds = scaler.inverse_transform(gru_preds.reshape(-1, scaled_data.shape[1]))
actual = scaler.inverse_transform(y_test.numpy().reshape(-1, scaled_data.shape[1]))

# Plot
print("Plotting predictions...")
plot_predictions(actual, lstm_preds, gru_preds, show=False)
print("Plot saved to outputs/plots/prediction.png")

# Future prediction
print("\nGenerating future predictions...")
future = predict_future(lstm, X_test[-1], FUTURE_STEPS)
# Reshape for inverse transform: future is (FUTURE_STEPS, 1, features) -> (FUTURE_STEPS, features)
future = future.reshape(-1, scaled_data.shape[1])
future = scaler.inverse_transform(future)

print("Next 7 days predicted prices:\n", future)