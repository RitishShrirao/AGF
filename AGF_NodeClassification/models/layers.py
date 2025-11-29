import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SoftmaxAttention, SVDAttention, HybridAttention

## Attention layer
class AttentionLayer(nn.Module):
    def __init__(self, max_seq_len, attn_type, d_model, n_heads, low_rank=20, rank_multi=10, nb_features=64,
                 attention_dropout=0.1, d_keys=None, d_values=None, poly_type=None, K=None, alpha=None, beta=None, fixI=None):
        super(AttentionLayer, self).__init__()
        self.d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.attn_type = attn_type
        self.max_seq_len = max_seq_len

        if self.attn_type == "softmax" or self.attn_type in ["svd", "hybrid"]:
            d_values = d_values or (d_model // n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.out_projection = nn.Linear(self.d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.low_rank =  low_rank
        self.rank_multi = rank_multi
        self.len = min(self.max_seq_len, self.low_rank * self.rank_multi)

        if self.attn_type == "softmax":
            self.inner_attention = SoftmaxAttention(self.d_keys, attention_dropout)
        elif self.attn_type == "svd":
            self.Sigma_projection = nn.Linear(d_model, self.d_keys * n_heads)
            self.inner_attention = SVDAttention(self.d_keys, attention_dropout, poly_type, K=K, alpha=alpha, beta=beta, fixI=fixI)
        elif self.attn_type == "hybrid":
            # For hybrid: split heads into two groups
            assert n_heads % 2 == 0, "Number of heads must be even for hybrid attention"
            self.n_heads_agf = n_heads // 2
            self.n_heads_softmax = n_heads // 2
            self.Sigma_projection = nn.Linear(d_model, self.d_keys * self.n_heads_agf)
            self.inner_attention = HybridAttention(self.d_keys, attention_dropout, poly_type, K=K, alpha=alpha, beta=beta, fixI=fixI)

    def forward(self, x):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)

        if self.attn_type == "softmax":
            values = self.value_projection(x).view(B, S, H, -1)
            out = self.inner_attention(queries, keys, values)
            out = out.reshape(B, L, -1)
            return self.out_projection(out)

        elif self.attn_type.startswith("svd"):
            values = self.value_projection(x).view(B, S, H, -1)
            Sigma = self.Sigma_projection(x).view(B, S, H, -1)
            out, ortho_loss = self.inner_attention(queries, Sigma, keys, values)
            out = out.reshape(B, L, -1)
            return self.out_projection(out), ortho_loss
        
        elif self.attn_type == "hybrid":
            values = self.value_projection(x).view(B, S, H, -1)
            Sigma = self.Sigma_projection(x).view(B, S, self.n_heads_agf, -1)
            
            # Split queries, keys, values for AGF and Flash heads
            queries_agf = queries[:, :, :self.n_heads_agf, :]
            keys_agf = keys[:, :, :self.n_heads_agf, :]
            values_agf = values[:, :, :self.n_heads_agf, :]
            
            queries_softmax = queries[:, :, self.n_heads_agf:, :]
            keys_softmax = keys[:, :, self.n_heads_agf:, :]
            values_softmax = values[:, :, self.n_heads_agf:, :]
            
            out, ortho_loss = self.inner_attention(queries_agf, Sigma, keys_agf, values_agf, 
                                                    queries_softmax, keys_softmax, values_softmax)
            out = out.reshape(B, L, -1)
            return self.out_projection(out), ortho_loss

class EncoderLayer(nn.Module):
    def __init__(self, attn_type, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attn_type = attn_type 
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        if self.attn_type == "softmax":
            new_x = self.attention(x)
        elif self.attn_type.startswith("svd") or self.attn_type == "hybrid":
            new_x, ortho_loss = self.attention(x)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if self.attn_type == "softmax":
            return self.norm2(x + y)
        elif self.attn_type.startswith("svd") or self.attn_type == "hybrid":
            return self.norm2(x + y), ortho_loss
        
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        score_list = []
        Lambda_list = []
        ortho_loss_list = []
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            if attn_layer.attention.inner_attention.__class__.__name__.startswith("Softmax"):
                x = attn_layer(x)
            elif attn_layer.attention.inner_attention.__class__.__name__.startswith("SVD") or \
                 attn_layer.attention.inner_attention.__class__.__name__.startswith("Hybrid"):
                x, ortho_loss = attn_layer(x)
                ortho_loss_list.append(ortho_loss)

        if self.norm is not None:
            x = self.norm(x)

        if len(ortho_loss_list) == 0:
            return x
        else:
            return x, ortho_loss_list
