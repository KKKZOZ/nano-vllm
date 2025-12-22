import os
import time

import torch
from transformers import AutoTokenizer

from nanovllm import SamplingParams
from nanovllm.backend import Backend
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.logger import logger


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = Backend(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=300)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "introduce yourself"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt)

    print(tokenizer.encode("请用中文回复"))

    temperature = torch.tensor(
        [sampling_params.temperature], dtype=torch.float32, pin_memory=True
    ).cuda(non_blocking=True)

    sampler = Sampler()

    output_ids = []

    logits, seq_id = llm.generate_v1(None, prompt_ids)

    token_id = sampler(logits, temperature).item()
    output_ids.append(token_id)

    t = time.perf_counter()
    while token_id != llm.eos and len(output_ids) < sampling_params.max_tokens:
        if len(output_ids) % 100 == 0:
            logger.info("Extend test, adding 4 tokens: [14880, 11622, 104811, 104787]")
            token_ids = [14880, 11622, 104811, 104787]
            output_ids.extend(token_ids)
            logits, _ = llm.generate_v1(seq_id, token_ids)
            token_id = sampler(logits, temperature).item()
            output_ids.append(token_id)
            continue

        logits, _ = llm.generate_v1(seq_id, [token_id])
        token_id = sampler(logits, temperature).item()
        output_ids.append(token_id)
    decode_time = time.perf_counter() - t
    llm.free_sequence(seq_id)

    logger.info(f"Decoding speed: {len(output_ids) / decode_time:.2f} tokens/s")
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output!r}")

    # output = llm.generate_v0(prompt, sampling_params)

    # print("\n")
    # print(f"Prompt: {prompt!r}")
    # print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
