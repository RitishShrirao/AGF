import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .polyconv import PolyConvFrame

## Canonical softmax attention
class SoftmaxAttention(nn.Module):
    def __init__(self, d_keys, attention_dropout):
        super(SoftmaxAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.d_keys = d_keys

    def forward(self, queries, keys, values):
        # queries: [B, L, H, D]
        queries = queries.transpose(1, 2) # [B, H, L, D]
        keys = keys.transpose(1, 2)       # [B, H, S, D]
        values = values.transpose(1, 2)   # [B, H, S, D]
        
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
