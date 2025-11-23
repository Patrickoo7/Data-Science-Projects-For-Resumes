"""
Time series forecasting models: LSTM, GRU, Transformer.
Advanced architectures with attention mechanisms.
"""

import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
    """LSTM-based stock price prediction model."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128, 64, 32],
        output_size: int = 1,
        dropout: float = 0.2,
        num_layers: int = 3
    ):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        layer_input_size = input_size

        for hidden_size in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(
                    layer_input_size,
                    hidden_size,
                    batch_first=True,
                    dropout=dropout if len(self.lstm_layers) < len(hidden_sizes) - 1 else 0
                )
            )
            layer_input_size = hidden_size

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Take the last time step
        x = x[:, -1, :]

        # Fully connected
        out = self.fc(x)
        return out


class GRUModel(nn.Module):
    """GRU-based stock price prediction model."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128, 64, 32],
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(GRUModel, self).__init__()

        self.gru_layers = nn.ModuleList()
        layer_input_size = input_size

        for hidden_size in hidden_sizes:
            self.gru_layers.append(
                nn.GRU(
                    layer_input_size,
                    hidden_size,
                    batch_first=True,
                    dropout=dropout if len(self.gru_layers) < len(hidden_sizes) - 1 else 0
                )
            )
            layer_input_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )

    def forward(self, x):
        for gru in self.gru_layers:
            x, _ = gru(x)

        x = x[:, -1, :]
        out = self.fc(x)
        return out


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for capturing temporal patterns."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # *2 for bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        out = self.fc(x)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    """Transformer-based time series prediction."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1
    ):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Take last time step
        x = x[:, -1, :]

        # Output
        out = self.fc(x)
        return out


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for interpretability."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x, return_attention=False):
        # LSTM output
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)

        # Attention weights
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        out = self.fc(context)

        if return_attention:
            return out, attention_weights
        return out


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for robust predictions."""

    def __init__(self, models: list, weights: list = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)

        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output


def create_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    **kwargs
):
    """Factory function to create time series models."""

    if model_type == 'lstm':
        return LSTMModel(input_size, output_size=output_size, **kwargs)
    elif model_type == 'gru':
        return GRUModel(input_size, output_size=output_size, **kwargs)
    elif model_type == 'bilstm':
        return BidirectionalLSTM(input_size, output_size=output_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerModel(input_size, output_size=output_size, **kwargs)
    elif model_type == 'attention_lstm':
        return AttentionLSTM(input_size, output_size=output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_len = 60
    input_size = 10

    dummy_input = torch.randn(batch_size, seq_len, input_size)

    print("Testing LSTM Model...")
    lstm_model = create_model('lstm', input_size)
    lstm_out = lstm_model(dummy_input)
    print(f"LSTM Output shape: {lstm_out.shape}")

    print("\nTesting Transformer Model...")
    transformer_model = create_model('transformer', input_size)
    transformer_out = transformer_model(dummy_input)
    print(f"Transformer Output shape: {transformer_out.shape}")

    print("\nTesting Attention LSTM...")
    attn_lstm = create_model('attention_lstm', input_size)
    attn_out, attn_weights = attn_lstm(dummy_input, return_attention=True)
    print(f"Attention LSTM Output shape: {attn_out.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    print("\nAll models tested successfully!")
