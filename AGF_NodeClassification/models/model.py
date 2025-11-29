import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Encoder, EncoderLayer, AttentionLayer

class AGFNodeClassifier(nn.Module):
    def __init__(self, num_features, num_classes, 
                 d_model=64, n_heads=4, num_layers=2, d_ff=256, 
                 dropout=0.1, attention_dropout=0.1, 
                 attn_type="svd", poly_type="jacobi", K=3, alpha=1.0, beta=0.2, fixI=False):
        super(AGFNodeClassifier, self).__init__()
        
        self.embedding = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(dropout)
        
        attn_layers = []
        for _ in range(num_layers):
            attn = AttentionLayer(
                max_seq_len=100000, # Large enough for node classification
                attn_type=attn_type,
                d_model=d_model,
                n_heads=n_heads,
                attention_dropout=attention_dropout,
                poly_type=poly_type,
                K=K,
                alpha=alpha,
                beta=beta,
                fixI=fixI
            )
            attn_layers.append(EncoderLayer(
                attn_type=attn_type,
                attention=attn,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout
            ))
            
        self.encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(d_model))
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [N, F]
        # We treat the whole graph as one sequence with batch size 1
        # x -> [1, N, F]
        x = x.unsqueeze(0)
        
        x = self.embedding(x)
        x = self.dropout(x)
        
        out = self.encoder(x)
        
        ortho_loss = 0
        if isinstance(out, tuple):
            x, ortho_losses = out
            if ortho_losses:
                ortho_loss = sum(ortho_losses) / len(ortho_losses)
        
        # x: [1, N, D] -> [N, D]
        x = x.squeeze(0)
        logits = self.classifier(x)
        
        return logits, ortho_loss
