import os
import torch
import numpy as np
import random

from train import train_valid
from test import test
from utils.data_utils import dna_to_onehot_extra
from models.oneDCNN import CNN1DRegressor
from models.Multitask import CNN1DMultitask
from models.oneDCNN import EnhancedCNN1DRegressor
from models.RNN import RNN_regressor
from models.i6mA_CapsNet import CapsNet
from models.CapsuleNetwork_own import CapsNetRegressor

seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If GPU usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Paths
forecast_lindel_train_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\combine_forecast_lindel_data\forecast_lindel_train.csv'
forecast_lindel_valid_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\combine_forecast_lindel_data\forecast_lindel_valid.csv'
forecast_train_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\forecast_train_val_test\train.csv'
forecast_valid_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\forecast_train_val_test\valid.csv'
forecast_test_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\forecast_train_val_test\test.csv'
lindel_train_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\compiled_lindel_data\lindel_train.csv'
lindel_valid_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\compiled_lindel_data\lindel_valid.csv'
lindel_test_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\compiled_lindel_data\lindel_test.csv'

model_save_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\saved_models'
model_save_name = '1DCNN_regressor.pth'
model_save_name_mt = 'Multitask_regressor.pth'
model_save_name_caps = 'CapsNet_regressor.pth'
model_save_file = os.path.join(model_save_path, model_save_name)
model_save_file_mt = os.path.join(model_save_path, model_save_name_mt)
model_save_file_caps = os.path.join(model_save_path, model_save_name_caps)

if __name__ == "__main__":
    set_seed(seed)
    print(f'Using device: {device}')

    # Select encoder and model
    #encoder = dna_to_onehot_extra(sequence_length=60)

    #model = CNN1DRegressor(input_channels=6).to(device)
    model = CapsNetRegressor().to(device)
    #model = CNN1DMultitask(input_channels=6).to(device)
    #model = RNN_regressor().to(device)

    # Paths
    train_path = forecast_lindel_train_path
    valid_path = forecast_lindel_valid_path

    # Run training
    print('Initializing Training...')

    train_valid(
        train_path=train_path,
        valid_path=valid_path,
        model_save_file=model_save_file_caps,
        model=model,
        epochs=50,
        batch_size=32,
        lr=0.01,
        patience=10
    )

    # Run testing
    print('Initializing Testing...')
    print('Testing' + model_save_name_caps)
    test([forecast_test_path, lindel_test_path], model_save_file_caps, model)
