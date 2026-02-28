import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

model_id = "/root/huggingface/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
).eval()

print(model.config._attn_implementation)

inputs = tok("你好，介绍一下你自己。", return_tensors="pt").to("cuda")

# warmup
with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
torch.cuda.synchronize()

start_time = time.perf_counter()
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
end_time = time.perf_counter()
print(tok.decode(out[0], skip_special_tokens=True))
out_len = out.shape[1]
print(f"Generated {out_len} tokens in {end_time - start_time:.2f} seconds")
print(f"Generation speed: {out_len / (end_time - start_time):.2f} tokens/second")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

model_id = "/home/kkkzoz/huggingface/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
).eval()

print(model.config._attn_implementation)

inputs = tok("你好，介绍一下你自己。", return_tensors="pt").to("cuda")

# warmup
with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
torch.cuda.synchronize()

start_time = time.perf_counter()
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=True)
end_time = time.perf_counter()
print(tok.decode(out[0], skip_special_tokens=True))
out_len = out.shape[1]
print(f"Generated {out_len} tokens in {end_time - start_time:.2f} seconds")
print(f"Generation speed: {out_len / (end_time - start_time):.2f} tokens/second")
