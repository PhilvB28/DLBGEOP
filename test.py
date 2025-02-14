#This File contains the testing function
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.eval import get_eval

from utils.data_utils import CRISPRDataset
from models.oneDCNN import CNN1DRegressor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Testing Function
def test(test_paths, model_save_file, model):
    # test_paths: [forecast_test_path, lindel_test_path]
    forecast_test_df = pd.read_csv(test_paths[0])
    lindel_test_df = pd.read_csv(test_paths[1])

    # Inspect DataFrames for debugging
    #print("Forecast Test DataFrame Head:")
    #print(forecast_test_df.head())
    #print("Lindel Test DataFrame Head:")
    #print(lindel_test_df.head())

    # Create datasets
    forecast_test_dataset = CRISPRDataset(forecast_test_df)
    lindel_test_dataset = CRISPRDataset(lindel_test_df)

    # Create data loaders
    test_loader = DataLoader(forecast_test_dataset, batch_size=32, shuffle=False)
    lindel_loader = DataLoader(lindel_test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(model_save_file, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Initialize lists to collect predictions and ground truths
    y_pred_forecast_list = []
    y_test_forecast_list = []
    y_pred_lindel_list = []
    y_lindel_list = []

    # Forecast Test Predictions
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_pred_forecast_list.append(outputs.cpu().numpy())
            y_test_forecast_list.append(y_batch.cpu().numpy())
    y_pred_forecast = np.concatenate(y_pred_forecast_list)
    y_forecast = np.concatenate(y_test_forecast_list)

    # Lindel Test Predictions
    with torch.no_grad():
        for X_batch, y_batch in lindel_loader:
            outputs = model(X_batch)
            y_pred_lindel_list.append(outputs.cpu().numpy())
            y_lindel_list.append(y_batch.cpu().numpy())
    y_pred_lindel = np.concatenate(y_pred_lindel_list, axis=0)
    y_lindel = np.concatenate(y_lindel_list, axis=0)

    # **Shape Checks** for debugging
    #print(f'y_test shape: {y_forecast.shape}')            # Should be (num_samples, 557)
    #print(f'y_pred shape: {y_pred_forecast.shape}')       # Should be (num_samples, 557)
    #print(f'y_lindel shape: {y_lindel.shape}')            # Should be (num_samples, 557)
    #print(f'y_pred_lindel shape: {y_pred_lindel.shape}')  # Should be (num_samples, 557)

    # General Evaluation
    forecast_eval_result = get_eval(y_pred_forecast, y_forecast)
    lindel_eval_result = get_eval(y_pred_lindel, y_lindel)

    # Show test results
    print('Prediction test values:')
    print(f'ForeCast MSE: {forecast_eval_result[0]:.8f}, ForeCast Pearson: {forecast_eval_result[1]:.8f}, ForeCast Spearman: {forecast_eval_result[2]:.8f}')
    print(f'Lindel MSE: {lindel_eval_result[0]:.8f}, Lindel Pearson: {lindel_eval_result[1]:.8f}, Lindel Spearman: {lindel_eval_result[2]:.8f}')