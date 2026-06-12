from dataclasses import fields
import torch.multiprocessing as mp
import atexit
from tqdm import tqdm
from time import perf_counter

from transformers import AutoTokenizer
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams

class LLMEngine:

    def __init__(self,model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config,0,self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast = True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
    
    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ):
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        
    def step(self):
        seqs, is_prefill = self.scheduler.schedule() #看到这了，基本没看schedule
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        

    def generate(
        self,
        prompts: list[str] |  list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    )-> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()