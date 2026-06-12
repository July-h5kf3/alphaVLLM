from collections import deque

from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(
        self,
        config: Config,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def is_finished(self):
        return not self.waiting and not self.running