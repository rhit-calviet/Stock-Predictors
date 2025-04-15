# evaluate.py
# Contains the evaluation loop function for the stock prediction model.

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm # For progress bar
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import necessary configurations
import config

def evaluate_model(model, dataloader, loss_fn, device, target_scaler):
    """
    Evaluates the multi-modal stock prediction model on a given dataset.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the evaluation data (e.g., test set).
        loss_fn (callable): The loss function (e.g., MSELoss).
        device (torch.device): The device to evaluate on (CPU or GPU).
        target_scaler (MinMaxScaler): The scaler used for the target variable,
                                      needed to inverse transform predictions.

    Returns:
        tuple: A tuple containing:
            - float: Average loss on the evaluation dataset.
            - float: Root Mean Squared Error (RMSE) in original price scale.
            - float: Mean Absolute Error (MAE) in original price scale.
            - float: Directional Accuracy (optional, simple version).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_predictions_scaled = []
    all_targets_scaled = []
    all_predictions_original = []
    all_targets_original = []

    print(f"Starting evaluation on {device}...")
    eval_progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad(): # Disable gradient calculations
        for batch in eval_progress_bar:
            stock_data = batch['stock_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets_scaled = batch['target'].to(device) # Scaled targets

            # --- Forward Pass ---
            predictions_scaled = model(stock_data, input_ids, attention_mask)

            # --- Calculate Loss ---
            loss = loss_fn(predictions_scaled, targets_scaled)
            total_loss += loss.item()

            # Store scaled predictions and targets for metric calculation
            all_predictions_scaled.extend(predictions_scaled.cpu().numpy())
            all_targets_scaled.extend(targets_scaled.cpu().numpy())

            eval_progress_bar.set_postfix(loss=loss.item())

    eval_progress_bar.close()
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Average Loss (Scaled): {avg_loss:.6f}")

    # --- Calculate Metrics in Original Scale ---
    predictions_scaled_np = np.array(all_predictions_scaled)
    targets_scaled_np = np.array(all_targets_scaled)

    # Inverse transform to get predictions and targets in original price scale
    predictions_original = target_scaler.inverse_transform(predictions_scaled_np)
    targets_original = target_scaler.inverse_transform(targets_scaled_np)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(targets_original, predictions_original))
    mae = mean_absolute_error(targets_original, predictions_original)

    print(f"Evaluation RMSE (Original Scale): {rmse:.4f}")
    print(f"Evaluation MAE (Original Scale): {mae:.4f}")

    # --- Optional: Calculate Directional Accuracy ---
    # Compares predicted direction (up/down) vs actual direction
    # Note: This requires comparing to the *previous day's actual close*
    # which is not directly available in this batch structure.
    # A simpler (less accurate) version compares consecutive target directions.
    actual_direction = np.sign(np.diff(targets_original.flatten()))
    predicted_direction = np.sign(np.diff(predictions_original.flatten()))
    # Ensure lengths match after diff()
    min_len = min(len(actual_direction), len(predicted_direction))
    directional_accuracy = np.mean(actual_direction[:min_len] == predicted_direction[:min_len]) * 100 if min_len > 0 else 0.0

    print(f"Evaluation Directional Accuracy (Approx.): {directional_accuracy:.2f}%")
    print("--- Evaluation Finished ---")

    return avg_loss, rmse, mae, directional_accuracy

