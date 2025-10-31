import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat

from distutils.version import LooseVersion

from .polyconv import PolyConvFrame

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

## Canonical softmax attention
class SoftmaxAttention(nn.Module):
    def __init__(self, d_keys, attention_dropout):
        super(SoftmaxAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.d_keys = d_keys

    def forward(self, queries, keys, values):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        dot = torch.matmul(queries, keys.transpose(-2, -1))
        dot = dot / math.sqrt(self.d_keys)

        attn = F.softmax(dot, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, values).transpose(1, 2).contiguous()

        return out

    
## SingularValueDecomposition attention
class SVDAttention(nn.Module):
    def __init__(self, d_keys, attention_dropout, poly_type=None, K=None, alpha=None, beta=None, fixI=None):
        super(SVDAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.d_keys = d_keys
        self.poly = PolyConvFrame(conv_fn_type=poly_type, depth=K, alpha=alpha, beta=beta, fixI=fixI)

    def forward(self, U, Sigma, V, values):
        
        U = nn.functional.softmax(U.transpose(1, 2), dim=-1)
        V_t = nn.functional.softmax(V.transpose(1, 2).transpose(-2, -1), dim=-1)

        values = values.transpose(1, 2)

        Sigma = Sigma.transpose(1,2)
        Sigma = nn.functional.sigmoid(Sigma)
        graph_filter = self.poly(Sigma)

        out = torch.matmul(U * graph_filter, torch.matmul(V_t, values)).transpose(1, 2).contiguous()

        # reg is eta in loss.py
        sym = torch.matmul(U.transpose(-2,-1), U)
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss = sym.abs().mean(dim=[1,2,3])

        sym = torch.matmul(V_t, V_t.transpose(-2,-1))
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss += sym.abs().mean(dim=[1,2,3])
        
        return out, ortho_loss


## Hybrid attention: half AGF (SVD) heads, half Flash Attention heads
class HybridAttention(nn.Module):
    def __init__(self, d_keys, attention_dropout, poly_type=None, K=None, alpha=None, beta=None, fixI=None):
        super(HybridAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.d_keys = d_keys
        self.poly = PolyConvFrame(conv_fn_type=poly_type, depth=K, alpha=alpha, beta=beta, fixI=fixI)

    def forward(self, U, Sigma, V, values, queries_softmax, keys_softmax, values_softmax):
        """
        First half of heads use AGF (SVD) attention
        Second half of heads use Flash Attention (torch's scaled_dot_product_attention)
        Args:
            U, Sigma, V: SVD components for AGF heads, input shape [B, L, H_agf, d_k]
            values: values for AGF heads, input shape [B, S, H_agf, d_k]
            queries_softmax, keys_softmax, values_softmax: Q, K, V for Flash heads [B, L/S, H_softmax, d_k]
        """
        # AGF attention for first half of heads
        # U: [B, L, H_agf, d_k] -> transpose(1,2) -> [B, H_agf, L, d_k]
        U = nn.functional.softmax(U.transpose(1, 2), dim=-1)  # [B, H_agf, L, d_k]
        # V: [B, S, H_agf, d_k] -> transpose(1,2) -> [B, H_agf, S, d_k] -> transpose(-2,-1) -> [B, H_agf, d_k, S]
        V_t = nn.functional.softmax(V.transpose(1, 2).transpose(-2, -1), dim=-1)  # [B, H_agf, d_k, S]

        # values: [B, S, H_agf, d_k] -> transpose(1,2) -> [B, H_agf, S, d_k]
        values_agf = values.transpose(1, 2)  # [B, H_agf, S, d_k]

        # Sigma: [B, S, H_agf, d_k] -> transpose(1,2) -> [B, H_agf, S, d_k]
        Sigma = Sigma.transpose(1,2)  # [B, H_agf, S, d_k]
        Sigma = nn.functional.sigmoid(Sigma)
        graph_filter = self.poly(Sigma)

        # U: [B, H_agf, L, d_k], V_t: [B, H_agf, d_k, S], values_agf: [B, H_agf, S, d_k]
        # V_t @ values_agf: [B, H_agf, d_k, S] @ [B, H_agf, S, d_k] -> [B, H_agf, d_k, d_k]
        # U * graph_filter: [B, H_agf, L, d_k] (element-wise)
        # (U * graph_filter) @ (V_t @ values_agf): [B, H_agf, L, d_k] @ [B, H_agf, d_k, d_k] -> [B, H_agf, L, d_k]
        # transpose(1,2) -> [B, L, H_agf, d_k]
        out_agf = torch.matmul(U * graph_filter, torch.matmul(V_t, values_agf)).transpose(1, 2).contiguous()  # [B, L, H_agf, d_k]

        # Orthogonality loss for AGF heads
        sym = torch.matmul(U.transpose(-2,-1), U)
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss = sym.abs().mean(dim=[1,2,3])

        sym = torch.matmul(V_t, V_t.transpose(-2,-1))
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss += sym.abs().mean(dim=[1,2,3])

        # Flash attention (torch's scaled_dot_product_attention) for second half of heads
        # Input: queries_softmax shape is [B, L, H_softmax, d_k]
        B, L, H_softmax, d_k = queries_softmax.shape
        
        # Transpose to [B, H_softmax, L, d_k] for scaled_dot_product_attention
        q = queries_softmax.permute(0, 2, 1, 3).contiguous()  # [B, H_softmax, L, d_k]
        k = keys_softmax.permute(0, 2, 1, 3).contiguous()  # [B, H_softmax, S, d_k]
        v = values_softmax.permute(0, 2, 1, 3).contiguous()  # [B, H_softmax, S, d_k]
        
        # SDPA expects [B, H, L, d_k]
        # Output: [B, H_softmax, L, d_k]
        out_softmax = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )  # [B, H_softmax, L, d_k]
        
        # Transpose back to [B, L, H_softmax, d_k] to match out_agf format
        out_softmax = out_softmax.permute(0, 2, 1, 3).contiguous()  # [B, L, H_softmax, d_k]

        # Concatenate outputs from both attention mechanisms along head dimension
        # out_agf: [B, L, H_agf, d_k], out_softmax: [B, L, H_softmax, d_k]
        # Final output: [B, L, H_total, d_k] to match SVDAttention output format
        out = torch.cat([out_agf, out_softmax], dim=2)  # [B, L, H_total, d_k]
        
        return out, ortho_loss


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
