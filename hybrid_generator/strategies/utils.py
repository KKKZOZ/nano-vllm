"""
Utility functions for hybrid generation.

This module contains helper functions for:
- Sampling (temperature, top-k, top-p, min-p)
- Uncertainty calculation (aleatoric and epistemic)
- Entropy calculation
"""

import time
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.cache_utils import DynamicCache


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a token from logits with temperature, top-k, top-p, and min-p filtering.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p
        min_p: Minimum probability threshold relative to the max probability

    Returns:
        tuple of (sampled_token, probabilities) where:
        - sampled_token: shape [batch_size, 1]
        - probabilities: shape [batch_size, vocab_size]
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Temperature = 0 means greedy
        return logits.argmax(dim=-1, keepdim=True), F.softmax(logits, dim=-1)

    # Apply min-p filtering
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        min_p_threshold = max_probs * min_p
        logits = torch.where(
            probs >= min_p_threshold,
            logits,
            torch.tensor(float("-inf")).to(logits.device),
        )

    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, top_k_indices, top_k_logits)

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1)

    return sampled_token, probs


def compute_logu(
    logits: torch.Tensor, topk: int = 10
) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculate log-u score of the prediction distribution.

    This computes aleatoric and epistemic uncertainty based on the
    logits distribution using the log-u method.

    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider

    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        Each is a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)

    # Get top-k logits and their indices
    topk_logits, _ = torch.topk(logits, topk, dim=-1)  # [batch_size, topk]

    # Calculate sum of logits (S)
    alpha = torch.sum(topk_logits, dim=-1, keepdim=True)  # [batch_size, 1]

    # Calculate normalized probabilities (p_i = x_i/S)
    probs = topk_logits / alpha  # [batch_size, topk]

    # Calculate digamma terms
    digamma_xi = torch.digamma(topk_logits + 1)  # ψ(x_i + 1)
    digamma_sum = torch.digamma(alpha + 1)  # ψ(S + 1)

    # Calculate aleatoric uncertainty efficiently
    # AU = -∑(p_i * (ψ(x_i + 1) - ψ(S + 1)))
    aleatoric_uncertainty = -torch.sum(
        probs * (digamma_xi - digamma_sum), dim=-1
    )  # [batch_size]

    # Calculate epistemic uncertainty
    # EU = K / (S + K)
    epistemic_uncertainty = topk / (alpha.squeeze(-1) + topk)  # [batch_size]

    if is_single_input:
        return aleatoric_uncertainty.item(), epistemic_uncertainty.item()
    else:
        return aleatoric_uncertainty, epistemic_uncertainty


def calculate_token_entropy(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Calculate token entropy from logits.

    Args:
        logits: Tensor of shape (vocab_size,) or (batch_size, vocab_size)
        temperature: Decoding temperature, default 1.0

    Returns:
        entropy: Scalar or Tensor of entropy values
    """
    # Handle temperature = 0 case (greedy decoding has zero entropy)
    if temperature == 0.0:
        if logits.dim() == 1:
            return torch.tensor(0.0, device=logits.device)
        else:
            return torch.zeros(logits.shape[0], device=logits.device)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Calculate probability distribution (Softmax)
    probs = F.softmax(scaled_logits, dim=-1)

    # Calculate entropy H = -sum(p * log(p))
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


# @torch.inference_mode()
# def simple_generate(
#     model_id: str,
#     prompt: str,
#     max_new_tokens: int = 1000,
#     device: str = "cuda",
# ):
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         attn_implementation="flash_attention_2",
#     ).to(device)

#     # 1. 预处理输入
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     input_ids = inputs["input_ids"]
#     prompt_len = input_ids.shape[1]

#     # 2. 初始化 StaticCache (预分配)
#     # max_cache_len 必须 >= prompt_len + max_new_tokens
#     max_cache_len = prompt_len + max_new_tokens
#     past_key_values = StaticCache(
#         config=model.config,
#         max_batch_size=1,
#         max_cache_len=max_cache_len,
#         device=device,
#         dtype=torch.float16,
#     )

#     # 3. Prefill 阶段 (独立执行)
#     print("Executing Prefill...")
#     cache_position = torch.arange(prompt_len, device=device)
#     out = model(
#         input_ids=input_ids,
#         past_key_values=past_key_values,
#         cache_position=cache_position,
#         use_cache=True,
#     )
#     next_token = out.logits[:, -1:].argmax(dim=-1)

#     # 4. 编译 Decode Step (可选，但推荐)
#     # 注意：torch.compile 对动态控制流支持有限，需要把 decode step 封装
#     def decode_one_step(input_ids, cache_pos):
#         return (
#             model(
#                 input_ids=input_ids,
#                 past_key_values=past_key_values,
#                 cache_position=cache_pos,
#                 use_cache=True,
#             )
#             .logits[:, -1:]
#             .argmax(dim=-1)
#         )

#     # decode_step_compiled = torch.compile(decode_one_step, mode="reduce-overhead")
#     # 如果不compile，StaticCache 也能带来巨大提升
#     decode_step = decode_one_step

#     generated_ids = [next_token.item()]
#     curr_pos = torch.tensor([prompt_len], device=device)  # 追踪当前位置

#     torch.cuda.synchronize()
#     t0 = time.time()

#     # 5. Decode Loop
#     for _ in range(max_new_tokens - 1):
#         # 传入当前 token 和位置
#         next_token = decode_step(next_token, curr_pos)

#         # 记录结果 (避免 GPU 上 torch.cat)
#         generated_ids.append(next_token.item())

#         # 更新位置
#         curr_pos += 1

#         # 简单的 EOS 检查 (放到 CPU 上做虽然慢，但比 GPU sync item 稍微好点，或者攒一批检查)
#         # 这里为了演示简单，依然使用 item()，但在 StaticCache 加持下 GPU 利用率会高很多
#         if generated_ids[-1] == tokenizer.eos_token_id:
#             break

#     torch.cuda.synchronize()
#     t1 = time.time()

#     speed = len(generated_ids) / (t1 - t0)
#     print(f"Decode Speed: {speed:.2f} tokens/s")

#     return tokenizer.decode(input_ids[0].tolist() + generated_ids)


@torch.inference_mode()
def simple_generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 2000,
    device: str = "cuda",
    dtype=torch.float16,
):
    print(f"Using model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Initialize cache
    past_key_values = DynamicCache(config=model.config)

    generated_ids = inputs["input_ids"]
    prompt_len = generated_ids.shape[1]

    # cache_position：prefill 阶段一般就是 [0..prompt_len-1]
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        eos_token_ids.extend(tokenizer.additional_special_tokens_ids)
    # print(f"Using eos_token_ids: {eos_token_ids}")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(max_new_tokens):
        outputs = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )

        next_token = outputs.logits[:, -1:].argmax(dim=-1)  # greedy
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() in eos_token_ids:
            break

        print(
            tokenizer.decode(next_token[0], skip_special_tokens=True),
            end="",
            flush=True,
        )

        inputs = {"input_ids": next_token}

        # cache_position 每轮 +1
        cache_position = cache_position[-1:] + 1

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    gen_tokens = generated_ids.shape[1] - prompt_len
    speed = gen_tokens / max(t1 - t0, 1e-9)
    print(
        f"\nTime taken: {t1 - t0:.2f}s, generated_tokens: {gen_tokens}, speed: {speed:.2f} tok/s"
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
