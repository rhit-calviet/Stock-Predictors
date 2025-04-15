# config.py
# Stores configuration settings and hyperparameters for the stock prediction model.

import torch
import datetime

# --- Data Configuration ---
STOCK_TICKER = 'NFLX'  # Example stock ticker
START_DATE = '2020-01-01' # Start date for fetching data
# Use today's date for end_date to get the latest data
END_DATE = datetime.date.today().strftime('%Y-%m-%d')
SEQUENCE_LENGTH = 365  # Use last 60 days of data to predict the next day
PREDICT_AHEAD = 1     # Predict 1 day ahead closing price

# --- Model Hyperparameters ---
FINBERT_MODEL_NAME = "ProsusAI/finbert" # Pre-trained FinBERT model from Hugging Face
EMBEDDING_DIM = 128  # Dimension for embedding the stock features
TRANSFORMER_NHEAD = 4  # Number of attention heads for the stock data transformer
TRANSFORMER_NUMLAYERS = 3  # Number of transformer encoder layers for stock data
TRANSFORMER_DIM_FEEDFORWARD = 256  # Hidden dimension in the transformer's feedforward network
FINBERT_OUTPUT_DIM = 768  # Output dimension of the chosen FinBERT model (BERT-base)
FUSION_DIM = 512  # Dimension of the hidden layer after fusing stock and news features
DROPOUT = 0.3     # Dropout rate for regularization

# --- Text Processing ---
MAX_LEN = 256     # Max sequence length for FinBERT tokenizer

# --- Training Hyperparameters ---
BATCH_SIZE = 4   # Batch size for training (adjust based on GPU memory)
LEARNING_RATE = 1e-5 # Learning rate for the AdamW optimizer
EPOCHS = 15        # Number of training epochs (start small for demonstration)
# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File Paths ---
MODEL_SAVE_PATH = f'complex_stock_predictor_{STOCK_TICKER}.pth' # Path to save the trained model

