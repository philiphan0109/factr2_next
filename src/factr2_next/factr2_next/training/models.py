import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        history,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        layers = [nn.Flatten()]
        width = input_size * history
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(width, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_size
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        num_layers=2,
        head_hidden=256,
        head_layers=2,
        bidirectional=False,
        dropout=0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_input = hidden_size * (2 if bidirectional else 1)
        self.head = regression_head(head_input, output_size, head_hidden, head_layers, dropout)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y[:, -1])


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        num_layers=2,
        head_hidden=256,
        head_layers=2,
        bidirectional=False,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_input = hidden_size * (2 if bidirectional else 1)
        self.head = regression_head(head_input, output_size, head_hidden, head_layers, dropout)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1])


def regression_head(input_size, output_size, head_hidden=256, head_layers=2, dropout=0.0):
    layers = []
    width = input_size
    for _ in range(max(0, int(head_layers) - 1)):
        layers.extend([nn.Linear(width, head_hidden), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = head_hidden
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


def build_model(model_cfg, input_size, output_size, history):
    model_type = str(model_cfg.get("type", "lstm")).lower()
    common = {
        "input_size": int(input_size),
        "output_size": int(output_size),
        "hidden_size": int(model_cfg.get("hidden_size", 128)),
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
    }
    if model_type == "mlp":
        return MLPRegressor(history=int(history), **common)
    if model_type == "gru":
        return GRURegressor(
            head_hidden=int(model_cfg.get("head_hidden", 256)),
            head_layers=int(model_cfg.get("head_layers", 2)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            **common,
        )
    if model_type == "lstm":
        return LSTMRegressor(
            head_hidden=int(model_cfg.get("head_hidden", 256)),
            head_layers=int(model_cfg.get("head_layers", 2)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            **common,
        )
    raise ValueError(f"Unsupported model type '{model_type}'. Use one of: mlp, gru, lstm.")
