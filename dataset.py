# dataset.py
# Defines the data preprocessing steps and the PyTorch Dataset class.

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer

# Import necessary configurations
import config

def preprocess_data(features_df, target_series):
    """
    Scales features and target variable, and creates sequences for model input.

    Args:
        features_df (pd.DataFrame): DataFrame containing numeric stock features.
        target_series (pd.Series): Series containing the target closing prices.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of stock feature sequences.
            - np.ndarray: Array of scaled target values.
            - np.ndarray: Array of placeholder news text sequences.
            - MinMaxScaler: Scaler used for features (to inverse transform later).
            - MinMaxScaler: Scaler used for the target variable.
    """
    print("Preprocessing data...")
    # Scale features
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features_df)

    # Scale target separately (important for inverse transform later)
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target_series.values.reshape(-1, 1))

    # Create sequences
    X_stock, y_target = [], []
    X_news_text = [] # Placeholder for news text

    print(f"Creating sequences of length {config.SEQUENCE_LENGTH}...")
    for i in range(len(features_scaled) - config.SEQUENCE_LENGTH):
        X_stock.append(features_scaled[i : i + config.SEQUENCE_LENGTH])
        # Target corresponds to the day AFTER the sequence ends
        y_target.append(target_scaled[i + config.SEQUENCE_LENGTH])

        # --- !!! Placeholder News Data Generation !!! ---
        # This section needs to be replaced with actual news fetching/loading logic.
        # Fetch news relevant to features_df.index[i + config.SEQUENCE_LENGTH - 1]
        date_of_news = features_df.index[i + config.SEQUENCE_LENGTH - 1]
        # Example placeholder based on simple price movement
        last_day_close = features_df['close'].iloc[i + config.SEQUENCE_LENGTH - 1]
        prev_day_close = features_df['close'].iloc[i + config.SEQUENCE_LENGTH - 2] if i + config.SEQUENCE_LENGTH > 1 else last_day_close
        movement = "rose" if last_day_close > prev_day_close else "fell"
        news_placeholder = (f"Stock price for {config.STOCK_TICKER} {movement} on "
                            f"{date_of_news.strftime('%Y-%m-%d')}. Market sentiment appears mixed. "
                            f"Volume was {features_df['volume'].iloc[i + config.SEQUENCE_LENGTH - 1]:.0f}.")
        X_news_text.append(news_placeholder)
        # --- End Placeholder ---

    X_stock = np.array(X_stock)
    y_target = np.array(y_target)
    X_news_text = np.array(X_news_text) # Keep as numpy array for now

    print(f"Created sequences: X_stock shape={X_stock.shape}, y_target shape={y_target.shape}, X_news shape={X_news_text.shape}")

    if X_stock.shape[0] == 0:
        raise ValueError("Not enough data to create sequences with the specified SEQUENCE_LENGTH.")

    return X_stock, y_target, X_news_text, feature_scaler, target_scaler


class StockNewsDataset(Dataset):
    """
    PyTorch Dataset for loading stock sequences, news text, and targets.
    Handles tokenization of news text.
    """
    def __init__(self, stock_sequences, news_texts, targets):
        """
        Args:
            stock_sequences (np.ndarray): Numpy array of stock feature sequences.
            news_texts (np.ndarray or list): Array or list of news text strings.
            targets (np.ndarray): Numpy array of scaled target values.
        """
        self.stock_sequences = torch.tensor(stock_sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.news_texts = list(news_texts) # Tokenizer expects list of strings
        print(f"Loading tokenizer: {config.FINBERT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)
        self.max_len = config.MAX_LEN # Max length for FinBERT tokenizer

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the stock sequence, tokenized news
                  (input_ids, attention_mask), and the target value.
        """
        stock_seq = self.stock_sequences[idx]
        target = self.targets[idx]
        news_text = self.news_texts[idx]

        # Tokenize news text for FinBERT
        encoding = self.tokenizer.encode_plus(
          news_text,
          add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
          max_length=self.max_len,      # Pad & truncate to max_len
          return_token_type_ids=False,  # Not needed for basic BERT/FinBERT
          padding='max_length',         # Pad to max_len
          truncation=True,              # Truncate if longer than max_len
          return_attention_mask=True,   # Return attention mask
          return_tensors='pt',          # Return PyTorch tensors
        )

        return {
          'stock_data': stock_seq,
          # Squeeze to remove the batch dimension added by return_tensors='pt'
          'input_ids': encoding['input_ids'].squeeze(0),
          'attention_mask': encoding['attention_mask'].squeeze(0),
          'target': target
        }

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create dummy data for testing
    print("Testing dataset functionality...")
    dummy_stock_features = pd.DataFrame(np.random.rand(200, 50), columns=[f'feat_{i}' for i in range(50)])
    dummy_stock_features['close'] = np.random.rand(200) * 100 + 100
    dummy_stock_features['volume'] = np.random.rand(200) * 1e6
    dummy_stock_features.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=200, freq='B')) # Business days
    dummy_target = dummy_stock_features['close'].shift(-config.PREDICT_AHEAD).dropna()
    dummy_stock_features = dummy_stock_features.loc[dummy_target.index]

    try:
        X_s, y_t, X_n, _, _ = preprocess_data(dummy_stock_features, dummy_target)

        # Test Dataset
        test_dataset = StockNewsDataset(X_s, X_n, y_t)
        print(f"Dataset size: {len(test_dataset)}")

        # Test DataLoader
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        sample_batch = next(iter(test_dataloader))

        print("\n--- Sample Batch ---")
        print("Stock Data Shape:", sample_batch['stock_data'].shape)
        print("Input IDs Shape:", sample_batch['input_ids'].shape)
        print("Attention Mask Shape:", sample_batch['attention_mask'].shape)
        print("Target Shape:", sample_batch['target'].shape)
        print("Sample Input IDs:", sample_batch['input_ids'][0, :15]) # Print first 15 token IDs

    except ValueError as e:
        print(f"Error during testing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

