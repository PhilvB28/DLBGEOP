#This File contains the training function
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm

from utils.data_utils import CRISPRDataset

from models.oneDCNN import CNN1DRegressor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_valid(train_path, valid_path, model_save_file, model, epochs=50, batch_size=16, lr=0.001, patience=5):
    # Load data
    #train_df = pd.read_csv(train_path, nrows=2000) #small dataset for debugging
    #valid_df = pd.read_csv(valid_path, nrows=500)

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # Create datasets
    train_dataset = CRISPRDataset(train_df)
    valid_dataset = CRISPRDataset(valid_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Prepare the model, loss function, and optimizer
    criterion = nn.BCELoss()  #MSELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize for early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_valid_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item() * X_batch.size(0)
        valid_loss /= len(valid_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.10f}, Valid Loss: {valid_loss:.10f}')

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the model
    torch.save(model.state_dict(), model_save_file)
    print("Training abgeschlossen und Modell gespeichert.")


def compute_loss(outputs, targets, criterion, multitask=False):
    """
    Compute the loss for single-task or multitask outputs.
    """
    if multitask:
        losses = [
            criterion(outputs[i], targets[i]) for i in range(len(outputs))
        ]
        return sum(losses)
    else:
        return criterion(outputs, targets)


def evaluate_model(model, data_loader, criterion, multitask=False):
    """
    Evaluate the model on a validation set.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            if multitask:
                loss = compute_loss(outputs, y_batch, criterion, multitask=True)
            else:
                loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(data_loader.dataset)
