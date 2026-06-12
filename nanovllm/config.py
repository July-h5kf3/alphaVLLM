import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass(slots=True)
class Config:
    model: str #模型目录路径
    max_num_batched_tokens: int = 16384 # 单次batch中允许处理的最大token数
    max_num_seqs: int = 512 #一个batch中最多同时处理的序列数目
    max_model_len: int = 4096 # 最长的seq len
    gpu_memory_utilization: float = 0.9 # 允许使用的显存比例
    tensor_parallel_size: int = 1 #张量并行路数
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256 #KV Cache Block的大小
    num_kvcache_blocks: int = -1 # KV Cache Block总数，-1表示自动估计

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)