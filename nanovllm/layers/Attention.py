import torch
from torch import nn

import triton
import triton.language as tl

from nanovllm.utils.context import get_context
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    N: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot != -1:
        key_block_ptr = tl.make_block_ptr(
            base=key_ptr,
            shape=(N, D),
            strides=(key_stride, 1),
            offsets=(idx, 0),
            block_shape=(1, D),
            order=(1, 0),
        )
        value_block_ptr = tl.make_block_ptr(
            base=value_ptr,
            shape=(N, D),
            strides=(value_stride, 1),
            offsets=(idx, 0),
            block_shape=(1, D),
            order=(1, 0),
        )
        k_cache_block_ptr = tl.make_block_ptr(
            base=k_cache_ptr,
            shape=(NUM_SLOTS, D),
            strides=(D, 1),
            offsets=(slot, 0),
            block_shape=(1, D),
            order=(1, 0),
        )
        v_cache_block_ptr = tl.make_block_ptr(
            base=v_cache_ptr,
            shape=(NUM_SLOTS, D),
            strides=(D, 1),
            offsets=(slot, 0),
            block_shape=(1, D),
            order=(1, 0),
        )

        key = tl.load(key_block_ptr, boundary_check=(0, 1))
        value = tl.load(value_block_ptr, boundary_check=(0, 1))
        tl.store(k_cache_block_ptr, key, boundary_check=(0, 1))
        tl.store(v_cache_block_ptr, value, boundary_check=(0, 1))

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert k_cache.numel() % D == 0 and v_cache.numel() % D == 0
    assert k_cache.numel() == v_cache.numel()
    assert slot_mapping.numel() == N
    num_slots = k_cache.numel() // D
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, N, num_slots, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
    
    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        v : torch.Tensor,
    ):
        context = get_context()
        k_cache,v_cache = self.k_cache,self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k,v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
