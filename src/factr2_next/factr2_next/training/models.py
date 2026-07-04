import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        num_layers=2,
        head_hidden=256,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head = [nn.Linear(hidden_size, head_hidden), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_hidden, output_size))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1])
