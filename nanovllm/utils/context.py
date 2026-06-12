from dataclasses import dataclass
import torch

#Context的作用主要是给Attention层传递当前batch的推理元信息

@dataclass(slots=True)
class Context:
    is_prefill: bool = False

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    slot_mapping: torch.Tensor | None = None #表示当前token的KV应该写入KV cache的哪个槽
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None #表示每条序列对应哪些KV cache block即paged KV cache的索引表

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()