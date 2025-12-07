import torch
import torch.nn as nn
import torch.nn.functional as F

class MitoAttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) model.
    
    Architecture:
    1. Feature Extractor: Projects raw features to a hidden dimension.
    2. Multihead Attention (Self-Attention): Captures relationships between mitochondria (instances).
    3. Pooling: Aggregates the sequence into a single latent vector using attention weights.
    4. Classifier: Predicts the class (CT vs ELA).
    """
    def __init__(self, input_dim, hidden_dim=64, nhead=4, dropout=0.1):
        super(MitoAttentionMIL, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Feature Extractor (Instance-level embedding)
        # Projects input features (e.g., 8) to hidden_dim (e.g., 64)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 2. Transformer Encoder Layer (Self-Attention)
        # We use a single layer for interpretability of heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # Binary classification (CT/ELA)
        )
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (Batch, Seq_Len, Features)
            lengths: (Batch) - Actual lengths of sequences
            
        Returns:
            logits: (Batch, 2)
            attn_weights: (Batch, Num_Heads, Seq_Len, Seq_Len)
            latent: (Batch, Hidden_Dim) - The vector before classification
        """
        # 1. Feature Extraction
        # x: (Batch, Seq, Features) -> (Batch, Seq, Hidden)
        h = self.feature_extractor(x)
        
        # 2. Attention Mask
        # Create mask for padding: (Batch, Seq) where True is padding
        key_padding_mask = None
        if lengths is not None:
            B, L, _ = x.shape
            # Create mask: True where index >= length
            key_padding_mask = torch.arange(L, device=x.device)[None, :] >= lengths[:, None]
            
        # 3. Transformer Self-Attention
        # attn_output: (Batch, Seq, Embed)
        # attn_weights: (Batch, Heads, Seq, Seq) (if average_attn_weights=False)
        attn_out, attn_weights = self.attention(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False 
        )
        
        # Residual + Norm
        h_trans = self.norm(h + attn_out)
        
        # 4. Pooling
        # We can take the mean of the non-padded elements
        if lengths is not None:
            # Mask padding
            mask = (~key_padding_mask).unsqueeze(-1).float() # (Batch, Seq, 1)
            sum_out = torch.sum(h_trans * mask, dim=1)
            count = torch.sum(mask, dim=1)
            latent = sum_out / count.clamp(min=1e-9)
        else:
            latent = torch.mean(h_trans, dim=1)
            
        # 5. Classifier
        logits = self.classifier(latent)
        
        return logits, attn_weights, latent
