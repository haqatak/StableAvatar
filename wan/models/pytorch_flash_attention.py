"""
PyTorch implementation of Flash Attention based on the MLX flashy-attention library.
This provides a drop-in replacement for CUDA flash attention with the same interface.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of flash attention.

    Args:
        q (torch.Tensor): The query vectors, in B x N x D.
        k (torch.Tensor): The key vectors, in B x M x D.
        v (torch.Tensor): The value vectors, in B x M x D_v.
        mask (torch.Tensor, optional): The attention mask. Should be broadcastable to (B, N, M).
        scale (float, optional): The scale factor for the attention scores.
        causal (bool, optional): If True, apply a causal mask.
        q_lens (torch.Tensor, optional): The sequence lengths of the queries, in B.
        k_lens (torch.Tensor, optional): The sequence lengths of the keys, in B.
        window_size (Tuple[int, int], optional): The size of the sliding window.

    Returns:
        (torch.Tensor, torch.Tensor): The output vectors and the log-sum-exp of attention scores.
    """
    B, N, D = q.shape
    _, M, D_v = v.shape
    device = q.device
    dtype = q.dtype

    if causal and N != M:
        raise ValueError("Causal masking requires query and key sequence lengths to be equal.")
    
    # Build combined mask
    combined_mask = None
    
    # Causal mask
    if causal:
        causal_mask = torch.triu(torch.full((N, M), float('-inf'), device=device, dtype=dtype), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # Expand to batch size
        combined_mask = causal_mask
    
    # Variable length mask
    if q_lens is not None and k_lens is not None:
        q_mask = torch.arange(N, device=device)[None, :] < q_lens[:, None]
        k_mask = torch.arange(M, device=device)[None, :] < k_lens[:, None]
        varlen_mask = q_mask[:, :, None] * k_mask[:, None, :]
        varlen_mask = torch.where(varlen_mask, 0.0, float('-inf'))
        
        if combined_mask is not None:
            combined_mask = combined_mask + varlen_mask
        else:
            combined_mask = varlen_mask
    
    # Sliding window mask
    if window_size[0] != -1 or window_size[1] != -1:
        query_indices = torch.arange(N, device=device).reshape(-1, 1)
        key_indices = torch.arange(M, device=device).reshape(1, -1)
        
        if window_size[0] == -1:
            lower_bound_mask = torch.ones((N, M), dtype=torch.bool, device=device)
        else:
            lower_bound_mask = key_indices >= (query_indices - window_size[0])
        
        if window_size[1] == -1:
            upper_bound_mask = torch.ones((N, M), dtype=torch.bool, device=device)
        else:
            upper_bound_mask = key_indices <= (query_indices + window_size[1])
        
        sliding_window_mask = lower_bound_mask & upper_bound_mask
        sliding_window_mask = torch.where(sliding_window_mask, 0.0, float('-inf'))
        sliding_window_mask = sliding_window_mask.unsqueeze(0).expand(B, -1, -1)  # Expand to batch size
        
        if combined_mask is not None:
            combined_mask = combined_mask + sliding_window_mask
        else:
            combined_mask = sliding_window_mask
    
    # Add custom mask
    if mask is not None:
        if combined_mask is not None:
            combined_mask = combined_mask + mask
        else:
            combined_mask = mask
    
    scale = scale or 1.0 / math.sqrt(D)
    
    # Online softmax statistics
    o = torch.zeros((B, N, D_v), dtype=dtype, device=device)
    l = torch.zeros((B, N), dtype=dtype, device=device)
    m = torch.full((B, N), float('-inf'), dtype=dtype, device=device)
    
    # Tiling parameters
    B_c = min(128, M)
    B_r = min(128, N)
    
    # Outer loop over keys and values
    for j in range(0, M, B_c):
        j_end = min(j + B_c, M)
        k_j = k[:, j:j_end, :]
        v_j = v[:, j:j_end, :]
        
        # Inner loop over queries
        for i in range(0, N, B_r):
            i_end = min(i + B_r, N)
            q_i = q[:, i:i_end, :]
            m_i = m[:, i:i_end]
            l_i = l[:, i:i_end]
            
            # Compute attention scores
            s_ij = torch.bmm(q_i, k_j.transpose(-2, -1)) * scale
            
            # Apply mask if present
            if combined_mask is not None:
                s_ij = s_ij + combined_mask[:, i:i_end, j:j_end]
            
            # Online softmax update
            m_i_new = torch.maximum(m_i, torch.max(s_ij, dim=-1)[0])
            p_ij = torch.exp(s_ij - m_i_new[:, :, None])
            
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(p_ij, dim=-1)
            
            # Update output
            o_i = o[:, i:i_end, :]
            o[:, i:i_end] = (l_i[:, :, None] / l_i_new[:, :, None]) * torch.exp(
                m_i - m_i_new
            )[:, :, None] * o_i + torch.bmm(p_ij, v_j)
            
            # Update stats
            m[:, i:i_end] = m_i_new
            l[:, i:i_end] = l_i_new
    
    L = m + torch.log(l)
    return o, L


def _flash_attention_varlen_single(q, k, v, scale, causal, window_size):
    """Helper function for variable-length flash attention on a single sequence."""
    N, D = q.shape
    M, D_v = v.shape
    device = q.device
    dtype = q.dtype
    
    mask = None
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.full((N, M), float('-inf'), device=device, dtype=dtype), diagonal=1)
    
    # Sliding window mask
    if window_size[0] != -1 or window_size[1] != -1:
        query_indices = torch.arange(N, device=device).reshape(-1, 1)
        key_indices = torch.arange(M, device=device).reshape(1, -1)
        
        if window_size[0] == -1:
            lower_bound_mask = torch.ones((N, M), dtype=torch.bool, device=device)
        else:
            lower_bound_mask = key_indices >= (query_indices - window_size[0])
        
        if window_size[1] == -1:
            upper_bound_mask = torch.ones((N, M), dtype=torch.bool, device=device)
        else:
            upper_bound_mask = key_indices <= (query_indices + window_size[1])
        
        sliding_window_mask = lower_bound_mask & upper_bound_mask
        sliding_window_mask = torch.where(sliding_window_mask, 0.0, float('-inf'))
        
        if mask is not None:
            mask = mask + sliding_window_mask
        else:
            mask = sliding_window_mask
    
    # Online softmax statistics
    o = torch.zeros((N, D_v), dtype=dtype, device=device)
    l = torch.zeros((N,), dtype=dtype, device=device)
    m = torch.full((N,), float('-inf'), dtype=dtype, device=device)
    
    B_c = min(128, M)
    B_r = min(128, N)
    
    for j in range(0, M, B_c):
        j_end = min(j + B_c, M)
        k_j = k[j:j_end, :]
        v_j = v[j:j_end, :]
        
        for i in range(0, N, B_r):
            i_end = min(i + B_r, N)
            q_i = q[i:i_end, :]
            m_i = m[i:i_end]
            l_i = l[i:i_end]
            
            s_ij = torch.mm(q_i, k_j.T) * scale
            
            if mask is not None:
                s_ij = s_ij + mask[i:i_end, j:j_end]
            
            m_i_new = torch.maximum(m_i, torch.max(s_ij, dim=-1)[0])
            p_ij = torch.exp(s_ij - m_i_new[:, None])
            
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.sum(p_ij, dim=-1)
            
            o_i = o[i:i_end, :]
            o[i:i_end] = (l_i[:, None] / l_i_new[:, None]) * torch.exp(m_i - m_i_new)[:, None] * o_i + torch.mm(p_ij, v_j)
            
            m[i:i_end] = m_i_new
            l[i:i_end] = l_i_new
    
    L = m + torch.log(l)
    return o, L


def flash_attention_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
):
    """
    Variable-length flash attention forward pass.
    
    Args:
        q: Packed queries [total_q_len, num_heads, head_dim]
        k: Packed keys [total_k_len, num_heads, head_dim]  
        v: Packed values [total_k_len, num_heads, head_dim]
        cu_seqlens_q: Cumulative query sequence lengths [batch_size + 1]
        cu_seqlens_k: Cumulative key sequence lengths [batch_size + 1]
        scale: Attention scale factor
        causal: Whether to apply causal masking
        window_size: Sliding window size
    
    Returns:
        torch.Tensor: Output tensor with same shape as q
    """
    B = len(cu_seqlens_q) - 1
    o = torch.zeros_like(q)
    
    for i in range(B):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i+1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_i = q[q_start:q_end]
        k_i = k[k_start:k_end]
        v_i = v[k_start:k_end]
        
        o_i, _ = _flash_attention_varlen_single(q_i, k_i, v_i, scale, causal, window_size)
        o[q_start:q_end] = o_i
    
    return o


class FlashAttention(nn.Module):
    """
    PyTorch implementation of multi-head flash attention.
    
    This module expects the queries, keys, and values to be already projected
    and split into heads.
    """
    
    def __init__(self, num_heads: int, head_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
        q_scale: float = 1.0,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            queries: Query tensor [B, N, num_heads, head_dim]
            keys: Key tensor [B, M, num_heads, head_dim]
            values: Value tensor [B, M, num_heads, head_dim]
            mask: Attention mask
            causal: Whether to apply causal masking
            q_lens: Query sequence lengths
            k_lens: Key sequence lengths
            q_scale: Query scaling factor
            window_size: Sliding window size
            deterministic: Whether to use deterministic mode
            
        Returns:
            torch.Tensor: Output tensor [B, N, num_heads * head_dim]
        """
        if deterministic:
            torch.manual_seed(0)
        
        B, N, _, _ = queries.shape
        _, M, _, _ = keys.shape
        
        # Transpose to (B, num_heads, N, head_dim)
        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)
        
        # Scale queries
        if q_scale != 1.0:
            q = q * q_scale
        
        # Reshape for flash attention
        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, M, self.head_dim)
        v = v.reshape(B * self.num_heads, M, self.head_dim)
        
        # Expand sequence lengths for all heads
        if q_lens is not None:
            q_lens = q_lens.unsqueeze(1).expand(B, self.num_heads).reshape(B * self.num_heads)
        
        if k_lens is not None:
            k_lens = k_lens.unsqueeze(1).expand(B, self.num_heads).reshape(B * self.num_heads)
        
        # Expand mask for all heads
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B, self.num_heads, N, M).reshape(B * self.num_heads, N, M)
        
        # Flash attention
        output, _ = flash_attention_forward(
            q, k, v,
            mask=mask,
            scale=1.0 / math.sqrt(self.head_dim),
            causal=causal,
            q_lens=q_lens,
            k_lens=k_lens,
            window_size=window_size,
        )
        
        # Reshape and project back
        output = output.reshape(B, self.num_heads, N, self.head_dim).transpose(1, 2)
        output = output.reshape(B, N, self.num_heads * self.head_dim)
        
        # Apply output projection and dropout
        return self.dropout(self.out_proj(output))


def pytorch_flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Drop-in replacement for the original flash_attention function in StableAvatar.
    
    This function matches the exact interface of the original implementation
    but uses PyTorch operations instead of CUDA flash attention.
    """
    # Input validation
    if not isinstance(q, torch.Tensor):
        raise ValueError("q must be a torch.Tensor")
    
    # Ensure tensors are on the same device
    device = q.device
    
    # Handle the case where we have 4D tensors [B, L, num_heads, head_dim]
    if q.dim() == 4:
        B, L, num_heads, head_dim = q.shape
        
        # Reshape to [B * num_heads, L, head_dim] for compatibility
        q_reshaped = q.transpose(1, 2).reshape(B * num_heads, L, head_dim)
        k_reshaped = k.transpose(1, 2).reshape(B * num_heads, k.shape[1], head_dim)
        v_reshaped = v.transpose(1, 2).reshape(B * num_heads, v.shape[1], head_dim)
        
        # Apply query scaling if provided
        if q_scale is not None:
            q_reshaped = q_reshaped * q_scale
        
        # Use the softmax scale or default
        scale = softmax_scale or 1.0 / math.sqrt(head_dim)
        
        # Expand sequence lengths for all heads
        if q_lens is not None:
            q_lens_expanded = q_lens.unsqueeze(1).expand(B, num_heads).reshape(B * num_heads)
        else:
            q_lens_expanded = None
            
        if k_lens is not None:
            k_lens_expanded = k_lens.unsqueeze(1).expand(B, num_heads).reshape(B * num_heads)
        else:
            k_lens_expanded = None
        
        # Run flash attention
        output, _ = flash_attention_forward(
            q_reshaped, k_reshaped, v_reshaped,
            mask=None,
            scale=scale,
            causal=causal,
            q_lens=q_lens_expanded,
            k_lens=k_lens_expanded,
            window_size=window_size,
        )
        
        # Reshape back to [B, L, num_heads, head_dim]
        output = output.reshape(B, num_heads, L, head_dim).transpose(1, 2)
        
        return output.to(dtype)
    
    else:
        # Handle other tensor shapes - fallback to standard attention
        warnings.warn("Using fallback attention for non-4D tensors")
        return F.scaled_dot_product_attention(
            q.transpose(-2, -3), 
            k.transpose(-2, -3), 
            v.transpose(-2, -3),
            is_causal=causal,
            dropout_p=dropout_p if deterministic else dropout_p
        ).transpose(-2, -3)
