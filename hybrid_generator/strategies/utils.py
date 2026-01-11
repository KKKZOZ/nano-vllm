"""
Utility functions for hybrid generation.

This module contains helper functions for:
- Sampling (temperature, top-k, top-p, min-p)
- Uncertainty calculation (aleatoric and epistemic)
- Entropy calculation
"""

import time
from typing import Tuple, Union

import flashinfer
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from nanovllm.utils.logger import logger


def _apply_sampling_filters(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        min_p_threshold = max_probs * min_p
        logits = torch.where(
            probs >= min_p_threshold,
            logits,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
        )

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, top_k_indices, top_k_logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def get_sampling_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Return the sampling probability distribution after applying filtering.
    """
    if temperature <= 0.0:
        return F.softmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / temperature

    logits = _apply_sampling_filters(logits, top_k, top_p, min_p)
    return F.softmax(logits, dim=-1)


@torch.compile()
def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Sample a token from logits with temperature, top-k, top-p, and min-p filtering.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p
        min_p: Minimum probability threshold relative to the max probability

    Returns:
        sampled_token: shape [batch_size]
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1).to(torch.int32)

    if temperature != 1.0:
        logits = logits / temperature

    logits = _apply_sampling_filters(logits, top_k, top_p, min_p)
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled_token.to(torch.int32)


@torch.compile(fullgraph=True)
def sample_token_optimized(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Optimized sampling:
    1. Fuses steps to reduce memory I/O.
    2. Avoids sorting the full vocabulary if Top-K is active.
    3. Calculates Softmax as late as possible.
    """
    # 1. Apply Temperature
    if abs(temperature - 1.0) > 1e-6:
        if temperature == 0.0:
            # Greedy decoding
            return logits.argmax(dim=-1).to(torch.int32)
        logits = logits / temperature

    # 2. Pre-calculate Softmax only if needed for Min-P
    # (Efficiency Trade-off: If min_p > 0, we must calc probabilities early)
    probs = None
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        # 直接在 logits 上操作，避免后续重复计算 softmax
        # 这里的 mask 逻辑可以融合
        logits = torch.where(
            probs >= (max_probs * min_p),
            logits,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
        )

    # 3. Efficient Top-K & Top-P
    # 关键优化：如果同时开启 Top-K 和 Top-P，先做 Top-K，
    # 然后在缩小的 K 个元素上做 Top-P，而不是对整个 vocab 排序。

    current_logits = logits
    current_indices = None  # None implies indices are [0, 1, ... V-1]

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # 只取 Top-K，后续所有计算只针对这 K 个值
        current_logits, current_indices = torch.topk(logits, top_k, dim=-1)

    if top_p < 1.0:
        # 如果前面做了 Top-K，这里的 current_logits 只有 K 个元素，排序非常快
        # 如果没做 Top-K，这里依然需要全量排序
        sorted_logits, sorted_indices = torch.sort(
            current_logits, descending=True, dim=-1
        )

        # 计算 Top-P 截断
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 确定保留的掩码
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # 至少保留一个

        # 将不需要的部分设为 -inf
        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove, float("-inf")
        )

        # 更新 current_logits
        current_logits = sorted_logits

        # 如果之前有 indices (即经过了 top-k)，需要映射回去
        if current_indices is not None:
            # sorted_indices 是相对于 top-k 结果的索引
            # current_indices 是 top-k 挑选出的原始 vocab 索引
            current_indices = torch.gather(current_indices, -1, sorted_indices)
        else:
            current_indices = sorted_indices

    # 4. Final Sampling
    # 此时 current_logits 可能只有 K 个元素，或者经过了 Top-P 过滤
    # 我们只对剩下的有效元素做 Softmax 和 Multinomial
    safe_probs = F.softmax(current_logits, dim=-1)
    sampled_index_in_subset = torch.multinomial(safe_probs, num_samples=1)

    # 5. Recover original index
    if current_indices is not None:
        sampled_token = torch.gather(current_indices, -1, sampled_index_in_subset)
    else:
        sampled_token = sampled_index_in_subset

    return sampled_token.squeeze(-1).to(torch.int32)


def sample_token_flashinfer(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    使用 FlashInfer 进行高性能采样 (Top-K + Top-P)。

    Args:
        logits: [Batch, Vocab], 推荐 float16 或 bfloat16
        temperature: 标量温度
        top_p: 标量 Top-P
        top_k: 标量 Top-K

    Returns:
        sampled_ids: [Batch], 采样得到的 Token ID
    """
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits with shape [batch, vocab], got {logits.shape}."
        )
    # logger.info(
    #     f"Sampling logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}"
    # )
    if logits.device.type != "cuda":
        logger.error("logits is not on CUDA device!")
        # sampled = sample_token(logits, temperature, top_k, top_p, 0.0)
        # return sampled.squeeze(-1).to(torch.int32)

    # if abs(temperature - 1.0) > 1e-6:
    #     if temperature == 0.0:
    #         return torch.argmax(logits, dim=-1).to(torch.int32)
    #     logits = logits / temperature

    logits = logits / temperature
    vocab_size = logits.shape[-1]
    if top_k <= 0 or top_k > vocab_size:
        top_k = vocab_size
    if top_p <= 0.0:
        return torch.argmax(logits, dim=-1).to(torch.int32)
    if top_p > 1.0:
        top_p = 1.0

    # FlashInfer API expects float32 logits and returns int32 samples.
    logits = logits.float()
    if not logits.is_contiguous():
        logits = logits.contiguous()

    sampled_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        deterministic=True,
    )

    return sampled_ids


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


# @torch.compile
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

    # dist = Categorical(logits=logits / temperature)
    # return dist.entropy()


# @torch.compile(mode="reduce-overhead")
def calculate_token_entropy_fast(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    # 1. 处理 Temperature
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # 2. 计算 LogSumExp (LSE) - 这是一个 Reduce 操作
    # keepdim=True 方便后续广播减法，虽然后面是点乘不需要
    lse = torch.logsumexp(logits, dim=-1)

    # 3. 计算 Softmax (Probs)
    # 注意：这里我们不需要显式计算 log_probs
    probs = torch.softmax(logits, dim=-1)

    # 4. 计算期望 E[z] = sum(p * z)
    expected_logits = torch.sum(probs * logits, dim=-1)

    # 5. 应用公式 H = LSE - E[z]
    entropy = lse - expected_logits

    return entropy


@torch.inference_mode()
def simple_generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 2000,
    device: str = "cuda",
    dtype=torch.float16,
    enable_stats_sync: bool = False,
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

    if enable_stats_sync and device.startswith("cuda"):
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

    if enable_stats_sync and device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    gen_tokens = generated_ids.shape[1] - prompt_len
    speed = gen_tokens / max(t1 - t0, 1e-9)
    print(
        f"\nTime taken: {t1 - t0:.2f}s, generated_tokens: {gen_tokens}, speed: {speed:.2f} tok/s"
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


@triton.jit
def _token_entropy_kernel(
    logits_ptr,
    entropy_ptr,
    vocab_size,
    temperature,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * vocab_size

    # Pass 1: Find max
    max_val = float("-inf")
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        max_val = tl.maximum(max_val, tl.max(scaled, axis=0))

    # Pass 2: Compute sum(exp)
    sum_exp = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature
        sum_exp += tl.sum(tl.exp(scaled - max_val), axis=0)

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: Compute entropy
    entropy = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(row_start + offsets, mask=mask, other=float("-inf"))
        scaled = logits / temperature

        shifted = scaled - max_val
        p = tl.exp(shifted) / sum_exp
        log_p = shifted - log_sum_exp

        entropy += tl.sum(tl.where(mask, -p * log_p, 0.0), axis=0)

    tl.store(entropy_ptr + row_idx, entropy)


def calculate_token_entropy_triton(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    # logger.info(f"Calculating token entropy with Triton, logits shape: {logits.shape}")
    squeeze_output = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True

    batch_size, vocab_size = logits.shape

    if temperature == 0.0:
        entropy = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
        return entropy.squeeze(0) if squeeze_output else entropy

    logits = logits.contiguous()
    entropy = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)

    BLOCK_SIZE = triton.next_power_of_2(min(vocab_size, 4096))

    _token_entropy_kernel[(batch_size,)](
        logits,
        entropy,
        vocab_size,
        temperature,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return entropy.squeeze(0) if squeeze_output else entropy
