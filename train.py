# train.py
# Contains the training loop function for the stock prediction model.

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bar

# Import necessary configurations
import config

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs):
    """
    Trains the multi-modal stock prediction model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        optimizer (Optimizer): The optimizer to use (e.g., AdamW).
        loss_fn (callable): The loss function (e.g., MSELoss).
        device (torch.device): The device to train on (CPU or GPU).
        epochs (int): The number of epochs to train for.

    Returns:
        list: A list containing the average training loss for each epoch.
        list: A list containing the average validation loss for each epoch.
    """
    train_losses = []
    val_losses = []

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for batch in train_progress_bar:
            # Move batch data to the configured device
            stock_data = batch['stock_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # --- Forward Pass ---
            predictions = model(stock_data, input_ids, attention_mask)

            # --- Calculate Loss ---
            loss = loss_fn(predictions, targets)

            # --- Backward Pass and Optimization ---
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients
            # Optional: Gradient Clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()      # Update model parameters

            total_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item()) # Update progress bar

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_progress_bar.close()

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad(): # Disable gradient calculations for validation
            for batch in val_progress_bar:
                stock_data = batch['stock_data'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['target'].to(device)

                predictions = model(stock_data, input_ids, attention_mask)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_progress_bar.close()

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("--- Training Finished ---")
    return train_losses, val_losses

