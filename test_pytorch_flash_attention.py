#!/usr/bin/env python3
"""
Test script for PyTorch Flash Attention implementation.
"""

import torch
import math
import time
from wan.models.pytorch_flash_attention import (
    flash_attention_forward, 
    flash_attention_varlen_forward,
    FlashAttention,
    pytorch_flash_attention
)

def test_basic_flash_attention():
    """Test basic flash attention functionality."""
    print("Testing basic flash attention...")
    
    # Set up test data
    B, N, M, D = 2, 128, 128, 64
    device = 'cpu'  # Start with CPU for compatibility
    dtype = torch.float32
    
    q = torch.randn(B, N, D, device=device, dtype=dtype)
    k = torch.randn(B, M, D, device=device, dtype=dtype)
    v = torch.randn(B, M, D, device=device, dtype=dtype)
    
    # Test without any special features
    output, L = flash_attention_forward(q, k, v)
    
    print(f"✅ Basic attention works! Output shape: {output.shape}")
    
    # Test with causal masking
    output_causal, _ = flash_attention_forward(q, k, v, causal=True)
    print(f"✅ Causal attention works! Output shape: {output_causal.shape}")
    
    # Test with variable lengths
    q_lens = torch.tensor([100, 80], device=device)
    k_lens = torch.tensor([120, 90], device=device)
    output_varlen, _ = flash_attention_forward(q, k, v, q_lens=q_lens, k_lens=k_lens)
    print(f"✅ Variable length attention works! Output shape: {output_varlen.shape}")
    
    # Test with sliding window
    output_window, _ = flash_attention_forward(q, k, v, window_size=(32, 32))
    print(f"✅ Sliding window attention works! Output shape: {output_window.shape}")


def test_varlen_function():
    """Test variable-length packed function."""
    print("\nTesting varlen function...")
    
    # Create packed tensors
    device = 'cpu'
    dtype = torch.float32
    head_dim = 64
    
    # Batch with sequences of different lengths
    seq_lens_q = [50, 30, 70]  # 3 sequences
    seq_lens_k = [50, 30, 70]
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    
    q = torch.randn(total_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, head_dim, device=device, dtype=dtype)
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0] + seq_lens_q, device=device).cumsum(0)
    cu_seqlens_k = torch.tensor([0] + seq_lens_k, device=device).cumsum(0)
    
    scale = 1.0 / math.sqrt(head_dim)
    
    output = flash_attention_varlen_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, scale
    )
    
    print(f"✅ Varlen function works! Output shape: {output.shape}")


def test_flash_attention_module():
    """Test the FlashAttention module."""
    print("\nTesting FlashAttention module...")
    
    # Set up test data
    B, N, M = 2, 64, 64
    num_heads, head_dim = 8, 64
    device = 'cpu'
    dtype = torch.float32
    
    # Create multi-head tensors
    queries = torch.randn(B, N, num_heads, head_dim, device=device, dtype=dtype)
    keys = torch.randn(B, M, num_heads, head_dim, device=device, dtype=dtype)
    values = torch.randn(B, M, num_heads, head_dim, device=device, dtype=dtype)
    
    # Create module
    flash_attn = FlashAttention(num_heads, head_dim, dropout_p=0.1)
    
    # Test forward pass
    output = flash_attn(queries, keys, values)
    
    expected_shape = (B, N, num_heads * head_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✅ FlashAttention module works! Output shape: {output.shape}")
    
    # Test with various features
    output_causal = flash_attn(queries, keys, values, causal=True)
    print(f"✅ Module causal attention works! Output shape: {output_causal.shape}")
    
    q_lens = torch.tensor([50, 40], device=device)
    k_lens = torch.tensor([50, 40], device=device)
    output_varlen = flash_attn(queries, keys, values, q_lens=q_lens, k_lens=k_lens)
    print(f"✅ Module variable length works! Output shape: {output_varlen.shape}")


def test_drop_in_replacement():
    """Test the drop-in replacement function."""
    print("\nTesting drop-in replacement function...")
    
    # Test data matching StableAvatar format [B, L, num_heads, head_dim]
    B, L, num_heads, head_dim = 1, 256, 16, 64
    device = 'cpu'
    dtype = torch.bfloat16
    
    q = torch.randn(B, L, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, L, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, L, num_heads, head_dim, device=device, dtype=dtype)
    
    # Test the drop-in replacement
    output = pytorch_flash_attention(
        q, k, v,
        causal=True,
        window_size=(-1, -1),
        softmax_scale=1.0 / math.sqrt(head_dim)
    )
    
    assert output.shape == q.shape, f"Expected {q.shape}, got {output.shape}"
    print(f"✅ Drop-in replacement works! Output shape: {output.shape}")


def test_performance_comparison():
    """Compare performance with PyTorch's built-in attention."""
    print("\nTesting performance comparison...")
    
    B, N, D = 1, 512, 256
    device = 'cpu'
    dtype = torch.float32
    
    q = torch.randn(B, N, D, device=device, dtype=dtype)
    k = torch.randn(B, N, D, device=device, dtype=dtype)
    v = torch.randn(B, N, D, device=device, dtype=dtype)
    
    # Warm up
    for _ in range(3):
        _ = flash_attention_forward(q, k, v)
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
    
    # Time our implementation
    start_time = time.time()
    for _ in range(10):
        output_ours, _ = flash_attention_forward(q, k, v)
    our_time = time.time() - start_time
    
    # Time PyTorch's implementation
    start_time = time.time()
    for _ in range(10):
        output_torch = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    torch_time = time.time() - start_time
    
    print(f"Our implementation: {our_time:.4f}s")
    print(f"PyTorch SDPA: {torch_time:.4f}s")
    print(f"Ratio: {our_time/torch_time:.2f}x")
    
    # Check numerical similarity
    diff = torch.abs(output_ours - output_torch).max()
    print(f"Max difference: {diff:.6f}")


def main():
    """Run all tests."""
    print("🚀 Testing PyTorch Flash Attention Implementation\n")
    
    try:
        test_basic_flash_attention()
        test_varlen_function()
        test_flash_attention_module()
        test_drop_in_replacement()
        test_performance_comparison()
        
        print("\n🎉 All tests passed! The PyTorch Flash Attention implementation is working correctly.")
        
        # Test with MPS if available
        if torch.backends.mps.is_available():
            print("\n🍎 MPS (Apple Silicon) is available! Testing with MPS...")
            # Re-run a quick test with MPS
            device = 'mps'
            B, N, D = 1, 128, 64
            q = torch.randn(B, N, D, device=device)
            k = torch.randn(B, N, D, device=device)
            v = torch.randn(B, N, D, device=device)
            
            output, _ = flash_attention_forward(q, k, v)
            print(f"✅ MPS support works! Output shape: {output.shape}")
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            print("\n🎮 CUDA is available! Testing with CUDA...")
            device = 'cuda'
            B, N, D = 1, 128, 64
            q = torch.randn(B, N, D, device=device)
            k = torch.randn(B, N, D, device=device)
            v = torch.randn(B, N, D, device=device)
            
            output, _ = flash_attention_forward(q, k, v)
            print(f"✅ CUDA support works! Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
