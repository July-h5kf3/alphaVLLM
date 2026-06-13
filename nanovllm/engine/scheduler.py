from collections import deque

from nanovllm.engine.sequence import Sequence,SequenceStatus
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
        #存放新请求和被抢占后需要重新prefill的请求
        self.running: deque[Sequence] = deque()
        #存放已经完成prefiil，正在进行逐token decode的请求
    
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def schedule(self) -> tuple[list[Sequence],bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        #prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens#表示当前batch还能塞多少
            if remaining == 0:
                break
            if not seq.block_table: #说明该请求还没有分配KV Cache block
                num_cached_blocks = self.block_manager.can_allocate(seq)#还剩多少个cache block可以分配
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 这里说的是如果当前batch剩余token不够塞下这个seq，并且本batch已经有别的seq了，就不塞它了，因为只允许batch里的第一个请求被截断prefill
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
        
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 如果这轮之后这个prompt全部prefiil完成就切换状态
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True
            # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):#如果当前seq追加新token时需要新的block，但没有足够的free block
                if self.running:
                    self.preempt(self.running.pop())  #就从running队尾抢占一些请求，释放他们的KV Cache，重新置为waiting
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq) #如果加入一个新的token后需要新分配一个block
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    def postprocess(self,seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        #生成新token后，首先保存prefix cache
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_block(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens: #chunked prefill
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)