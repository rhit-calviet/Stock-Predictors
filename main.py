# main.py
# Main script to run the stock prediction pipeline.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # For plotting loss

# Import components from other files
import config
from data_utils import fetch_stock_data
from dataset import preprocess_data, StockNewsDataset
from model import ComplexStockPredictor
from train import train_model
from evaluate import evaluate_model # Import the evaluation function

def plot_loss(train_losses, val_losses, stock_ticker):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {stock_ticker}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_{stock_ticker}.png') # Save the plot
    print(f"Loss curve saved to loss_curve_{stock_ticker}.png")
    # plt.show() # Optionally display the plot

def run_pipeline():
    """Executes the complete data fetching, preprocessing, training, and evaluation pipeline."""
    print(f"Using device: {config.DEVICE}")

    # --- 1. Load and Prepare Data ---
    try:
        features_df, target_series = fetch_stock_data(config.STOCK_TICKER, config.START_DATE, config.END_DATE)
        num_stock_features = features_df.shape[1] # Get number of features dynamically

        X_stock, y_target, X_news, feature_scaler, target_scaler = preprocess_data(features_df, target_series)

        # Create the full dataset
        full_dataset = StockNewsDataset(X_stock, X_news, y_target)
        print(f"Full dataset size: {len(full_dataset)}")

    except ValueError as e:
        print(f"Data preparation error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Split Data (Train/Validation/Test) ---
    # Define split sizes
    test_split_size = int(len(full_dataset) * 0.15) # 15% for testing
    val_split_size = int(len(full_dataset) * 0.15) # 15% for validation
    train_split_size = len(full_dataset) - val_split_size - test_split_size

    if train_split_size <= 0 or val_split_size <= 0 or test_split_size <= 0:
        print("Error: Not enough data for train/validation/test split with current configuration.")
        print(f"Dataset size: {len(full_dataset)}, Train: {train_split_size}, Val: {val_split_size}, Test: {test_split_size}")
        return

    print(f"Splitting data: Train={train_split_size}, Validation={val_split_size}, Test={test_split_size}")
    # Perform chronological split (IMPORTANT for time series)
    # Ensure indices are aligned if using random_split (not recommended for time series)
    # Manual split is safer:
    train_indices = list(range(train_split_size))
    val_indices = list(range(train_split_size, train_split_size + val_split_size))
    test_indices = list(range(train_split_size + val_split_size, len(full_dataset)))

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # Shuffle=False for time series
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 3. Initialize Model, Optimizer, Loss ---
    try:
        model = ComplexStockPredictor(num_stock_features=num_stock_features).to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = nn.MSELoss() # Mean Squared Error for regression
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Train the Model ---
    try:
        train_losses, val_losses = train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_fn,
            config.DEVICE,
            config.EPOCHS
        )
        # Plot training/validation loss
        plot_loss(train_losses, val_losses, config.STOCK_TICKER)

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        # Optionally save model even if training interrupted early
        # torch.save(model.state_dict(), f'interrupted_{config.MODEL_SAVE_PATH}')
        return # Stop execution if training fails

    # --- 5. Evaluate the Model on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    try:
        test_loss, test_rmse, test_mae, test_dir_acc = evaluate_model(
            model,
            test_dataloader,
            loss_fn,
            config.DEVICE,
            target_scaler # Pass the scaler for inverse transformation
        )
        print(f"\nTest Set Performance:")
        print(f"  Average Loss: {test_loss:.6f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  Directional Accuracy (Approx.): {test_dir_acc:.2f}%")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

    # --- 6. Save the Trained Model ---
    try:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Model state dictionary saved to {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    run_pipeline()
