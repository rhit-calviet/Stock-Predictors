# model.py
# Defines the Positional Encoding and the main multi-modal Transformer model.

import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoModel

# Import necessary configurations
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class ComplexStockPredictor(nn.Module):
    def __init__(self, num_stock_features):
        super(ComplexStockPredictor, self).__init__()

        self.stock_embedding = nn.Linear(num_stock_features, config.EMBEDDING_DIM)
        self.embedding_dropout = nn.Dropout(config.DROPOUT)
        self.pos_encoder = PositionalEncoding(
            config.EMBEDDING_DIM, config.DROPOUT, max_len=config.SEQUENCE_LENGTH
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.TRANSFORMER_NHEAD,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=config.TRANSFORMER_NUMLAYERS
        )

        self.finbert = AutoModel.from_pretrained(config.FINBERT_MODEL_NAME)

        combined_dim = config.EMBEDDING_DIM + config.FINBERT_OUTPUT_DIM
        self.fusion_layer1 = nn.Linear(combined_dim, config.FUSION_DIM)
        self.fusion_dropout = nn.Dropout(config.DROPOUT)
        self.fusion_activation = nn.GELU()
        self.fusion_norm = nn.LayerNorm(config.FUSION_DIM)
        self.prediction_layer = nn.Linear(config.FUSION_DIM, 1)

    def forward(self, stock_data, input_ids, attention_mask):
        stock_embedded = self.stock_embedding(stock_data) * math.sqrt(config.EMBEDDING_DIM)
        stock_embedded = self.embedding_dropout(stock_embedded)
        stock_embedded = self.pos_encoder(stock_embedded)
        transformer_output = self.transformer_encoder(stock_embedded)
        stock_features = transformer_output.mean(dim=1)

        with torch.no_grad():
            outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        news_features = outputs.last_hidden_state.mean(dim=1)

        combined_features = torch.cat((stock_features, news_features), dim=1)
        fused = self.fusion_layer1(combined_features)
        fused = self.fusion_dropout(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_norm(fused)
        prediction = self.prediction_layer(fused)

        return prediction


# Example usage (optional, for testing)
if __name__ == '__main__':
    print("Testing model initialization...")
    # Dummy parameters
    num_features = 80 # Example number of stock features
    batch_s = 4
    seq_l = config.SEQUENCE_LENGTH
    max_l = config.MAX_LEN

    # Create dummy input tensors
    dummy_stock = torch.randn(batch_s, seq_l, num_features).to(config.DEVICE)
    dummy_ids = torch.randint(0, 30000, (batch_s, max_l)).to(config.DEVICE) # Vocab size approx 30k
    dummy_mask = torch.ones(batch_s, max_l).to(config.DEVICE)

    # Initialize model
    try:
        test_model = ComplexStockPredictor(num_stock_features=num_features).to(config.DEVICE)
        print("\nModel Architecture:")
        print(test_model)

        # Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            output = test_model(dummy_stock, dummy_ids, dummy_mask)
        print("Output Shape:", output.shape) # Expected: [batch_s, 1]
        assert output.shape == (batch_s, 1)
        print("Model forward pass successful.")

    except Exception as e:
        print(f"An error occurred during model testing: {e}")
        import traceback
        traceback.print_exc()

