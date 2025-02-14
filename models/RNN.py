import torch
import torch.nn as nn


class RNN_regressor(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_classes=557, num_layers=2, dropout=0.3, rnn_type="LSTM"):
        """
        Initializes an RNN-based model for CRISPR repair outcome prediction.

        Args:
            input_size (int): Dimensionality of the input embeddings (e.g., GloVe vector size).
            hidden_size (int): Number of hidden units in the RNN layers.
            num_classes (int): Number of output labels (e.g., 557 repair outcomes).
            num_layers (int): Number of RNN layers (default=2).
            dropout (float): Dropout rate for regularization (default=0.3).
            rnn_type (str): Type of RNN ('LSTM' or 'GRU', default='LSTM').
        """
        super(RNN_regressor, self).__init__()

        self.rnn_type = rnn_type

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Invalid rnn_type. Choose 'LSTM' or 'GRU'.")

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),  # Predict continuous values
        )

    def forward(self, x):
        if self.rnn_type == "LSTM":
            rnn_out, _ = self.rnn(x)  # Output: (batch_size, seq_len, hidden_size)
        else:
            rnn_out, _ = self.rnn(x)

        last_hidden_state = rnn_out[:, -1, :]
        output = self.fc(last_hidden_state)   # (batch_size, num_classes)
        return output