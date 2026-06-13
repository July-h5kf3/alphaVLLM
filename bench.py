import os
import time
from argparse import ArgumentParser
from random import randint, seed

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from nanovllm import LLM as LLM_nano
from nanovllm import SamplingParams as SamplingParams_nano
from vllm import LLM
from vllm import SamplingParams as SamplingParams_vllm

def main_nanovllm(prompt_token_ids, output_lens, path):
    llm = LLM_nano(path, enforce_eager=False, max_model_len=4096)
    sampling_params = [
        SamplingParams_nano(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
        for max_tokens in output_lens
    ]
    llm.generate(["Benchmark: "], SamplingParams_nano())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

def main_vllm(prompt_token_ids, output_lens, path):
    llm = LLM(
        model=path,
        enforce_eager=False,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=True,
    )
    prompts = [
        {"prompt_token_ids":ids}
        for ids in prompt_token_ids
    ]
    sampling_params = [
        SamplingParams_vllm(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
        for max_tokens in output_lens
    ]
    llm.generate(
        [{"prompt_token_ids":[1, 2, 3]}],
        SamplingParams_vllm(),
        use_tqdm=False,
    )
    t = time.time()
    llm.generate(prompts, sampling_params,use_tqdm=True)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("impl", choices=["nano", "vllm"], nargs="?", default="vllm")
    args = parser.parse_args()

    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    output_lens = [randint(100, max_ouput_len) for _ in range(num_seqs)]
    
    if args.impl == "nano":
        main_nanovllm(prompt_token_ids=prompt_token_ids, output_lens=output_lens, path=path)
    else:
        main_vllm(prompt_token_ids=prompt_token_ids, output_lens=output_lens, path=path)
