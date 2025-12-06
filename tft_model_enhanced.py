import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for time series prediction
    
    TASK: Predict price slope direction over a configurable time window
    - Class 0 (mapped from -1): Negative slope (price decreasing)
    - Class 1 (mapped from 0): Neutral slope (price flat)
    - Class 2 (mapped from 1): Positive slope (price increasing)
    
    Architecture:
    1. Input embedding and normalization
    2. Enhanced Variable Selection Network (VSN)
    3. LSTM for temporal encoding
    4. Temporal Fusion Decoder with attention
    5. Gated residual networks for enrichment
    6. Multi-head self-attention layers
    7. Feed-forward network
    8. Output layer for 3-class classification
    """
    def __init__(self, 
                 num_features: int,
                 hidden_size: int = 128,
                 lstm_layers: int = 3,
                 num_attention_heads: int = 8,
                 dropout: float = 0.3,
                 ffn_hidden_size: int = 256,
                 num_classes: int = 3):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced Variable Selection Network
        self.vsn = EnhancedVariableSelectionNetwork(
            num_features, 
            hidden_size, 
            dropout
        )
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        
        # Temporal Fusion Decoder
        self.temporal_fusion = TemporalFusionDecoder(
            hidden_size,
            num_attention_heads,
            dropout
        )
        
        # Gated Residual Networks for enrichment
        self.enrichment = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, dropout)
            for _ in range(2)
        ])
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(2)
        ])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # Output gating
        self.output_gate = GatedLinearUnit(hidden_size, dropout)
        
        # Output layer for classification
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        self._print_parameter_count()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def _print_parameter_count(self):
        """Print model parameter statistics"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} (trainable: {trainable:,})")
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Handle NaN/Inf values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        x_selected = self.vsn(x)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x_selected)
        
        # Temporal fusion
        fused = self.temporal_fusion(lstm_out, x_selected)
        
        # Enrichment through gated residual networks
        enriched = fused
        for grn in self.enrichment:
            enriched = grn(enriched)
        
        # Multi-head self-attention
        attn_out = enriched
        for attn, norm in zip(self.attention_layers, self.attention_norm):
            residual = attn_out
            attn_output, _ = attn(attn_out, attn_out, attn_out)
            attn_out = norm(residual + self.dropout(attn_output))
        
        # Feed-forward network
        residual = attn_out
        ffn_out = self.ffn(attn_out)
        attn_out = self.ffn_norm(residual + ffn_out)
        
        # Extract last timestep
        last_output = attn_out[:, -1, :]
        
        # Output gating
        gated_output = self.output_gate(last_output)
        
        # Classification output
        output = self.output_layer(gated_output)
        
        # Handle NaN in output
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class EnhancedVariableSelectionNetwork(nn.Module):
    """
    Enhanced Variable Selection Network (VSN)
    
    Learns importance weights for each input feature and applies
    context-aware feature transformation.
    """
    def __init__(self, num_features, hidden_size, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature transformation
        self.feature_transform = nn.Linear(num_features, hidden_size)
        
        # Context-aware gated residual network
        self.context_grn = GatedResidualNetwork(num_features, hidden_size, dropout)
        
        # Feature importance scoring
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_features)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_features)
        Returns:
            transformed: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten for processing
        x_flat = x.reshape(-1, self.num_features)
        
        # Learn context
        context = self.context_grn(x_flat)
        
        # Compute feature importance
        attention_scores = self.feature_attention(context)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply feature selection
        weighted_features = x_flat * attention_weights
        
        # Transform to hidden space
        transformed = self.feature_transform(weighted_features)
        transformed = self.layer_norm(transformed)
        transformed = self.dropout(transformed)
        
        # Reshape back
        output = transformed.reshape(batch_size, seq_len, self.hidden_size)
        
        return output


class TemporalFusionDecoder(nn.Module):
    """
    Temporal Fusion Decoder
    
    Fuses LSTM output with variable selection network output using
    multi-head attention and gating mechanisms.
    """
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.gate = GatedLinearUnit(hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, lstm_output, vsn_output):
        """
        Args:
            lstm_output: Output from LSTM
            vsn_output: Output from VSN
        Returns:
            Fused representation
        """
        attn_out, _ = self.attention(lstm_output, vsn_output, vsn_output)
        gated = self.gate(attn_out + lstm_output)
        return self.norm(gated)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU)
    
    Applies sigmoid gating to linear transformations for adaptive
    information flow control.
    """
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc(x)
        value, gate = x.chunk(2, dim=-1)
        return self.dropout(value * torch.sigmoid(gate))


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    
    Applies gated skip connections for flexible information flow
    and gradient propagation.
    """
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.skip_connection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        gate = torch.sigmoid(self.gate(F.gelu(self.fc1(x))))
        residual = self.skip_connection(x) if self.skip_connection is not None else x
        output = gate * out + (1 - gate) * residual
        return self.layer_norm(output)


class TradingDataset(Dataset):
    """
    Dataset for trading sequences
    
    MODIFIED: Now loads slope-based targets
    - Targets are mapped: -1 -> 0, 0 -> 1, 1 -> 2 for PyTorch compatibility
    """
    def __init__(self, X_path: str, y_path: str, mmap_mode: bool = True):
        if mmap_mode:
            self.X = np.load(X_path, mmap_mode='r')
            self.y = np.load(y_path, mmap_mode='r')
        else:
            self.X = np.load(X_path)
            self.y = np.load(y_path)
        
        # Convert labels from {-1, 0, 1} to {0, 1, 2}
        self.y = self.y + 1
        
        print(f"Dataset loaded: {len(self.X):,} samples")
        print(f"Memory mapping: {mmap_mode}")
        print(f"Sequence shape: {self.X.shape}")
        print(f"Target classes: {np.unique(self.y)}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])[0]
        return X, y


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model():
    """Test model initialization and forward pass"""
    from config import Config
    config = Config()
    num_features = config.get_num_features()
    
    print("="*60)
    print("TESTING TFT MODEL")
    print("="*60)
    
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        num_attention_heads=config.ATTENTION_HEADS,
        dropout=config.DROPOUT,
        ffn_hidden_size=config.FFN_HIDDEN_SIZE
    )
    
    # Test forward pass
    batch = torch.randn(16, config.LOOKBACK_WINDOW, num_features)
    output = model(batch)
    
    print(f"\nTest Results:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    total, trainable = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("="*60)
    print("✓ Model test successful!")


if __name__ == "__main__":
    test_model()