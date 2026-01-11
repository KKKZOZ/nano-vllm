import time
import json

import torch
from transformers import AutoTokenizer

# from generate import (
#     simple_generate,
# )
from hybrid_generator import HybridGenerator

input = "Let $O(0,0), A(\\tfrac{1}{2}, 0),$ and $B(0, \\tfrac{\\sqrt{3}}{2})$ be points in the coordinate plane. Let $\\mathcal{F}$ be the family of segments $\\overline{PQ}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\\overline{AB}$, distinct from $A$ and $B$, that does not belong to any segment from $\\mathcal{F}$ other than $\\overline{AB}$. Then $OC^2 = \\tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p + q$."
# input = "write a simple calculator in python"
llm = "/root/huggingface/Qwen3-8B"
slm = "/root/huggingface/Qwen3-1.7B"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": input},
]

tokenizer = AutoTokenizer.from_pretrained(llm)
input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# result = simple_generate("/root/huggingface/Qwen3-0.6B", input, max_new_tokens=1000)

# result = simple_generate("/root/huggingface/Qwen3-1.7B", input, max_new_tokens=1000)

# result = simple_generate("/root/huggingface/Qwen3-8B", input, max_new_tokens=1000)

# pipe = pipeline(
#     "text-generation",
#     model="/root/huggingface/Qwen3-8B",
#     torch_dtype=torch.float16,
#     device_map="auto",  # 自动分配 GPU
# )

# # 直接调用
# sequences = pipe(
#     "Hello, how are you?",
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=pipe.tokenizer.eos_token_id,
#     max_length=1000,
# )

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

# result = simple_generate_with_kv_cache_hf(model, input, max_new_tokens=20000)
# print(result)

# Greedy decoding (original version)
# print("Speculative decoding with greedy...")
# result = speculative_generate_with_kv_cache_hf(
#     draft_model_id=draft_model,
#     target_model_id=model,
#     prompt=input,
#     num_drafts=4,
#     max_new_tokens=1000,
# )

# # Sampling version with temperature, top-k, top-p, min-p
# print("Speculative decoding with sampling...")
# result = speculative_generate_with_sampling(
#     draft_model_id=draft_model,
#     target_model_id=model,
#     prompt=input,
#     num_drafts=4,
#     max_new_tokens=1000,
#     temperature=0.6,
#     top_k=20,
#     top_p=0.95,
#     min_p=0.0,
# )


def run_hybrid_generation(
    prompt,
    draft_model,
    model,
    strategy="entropy",
    slm_memory_usage=0.05,
    llm_memory_usage=0.9,
    threshold=0.1,
    max_new_tokens=1000,
    verbose=False,
):
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=model,
        slm_memory_usage=slm_memory_usage,
        llm_memory_usage=llm_memory_usage,
        device="cuda",
        dtype=torch.float16,
        verbose=verbose,
        enable_stats_sync=True,
    )
    time.sleep(600)
    result, stats = generator.generate(
        prompt=input,
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        threshold=threshold,
    )
    # print(f"Statatics: {stats}")

    engine_stats = generator.report_backend_stats()
    # print(f"Engine Backend Statistics: {engine_stats}")
    print(
        f"Engine Backend Statistics:\n{json.dumps(engine_stats, indent=2, ensure_ascii=False)}"
    )

    with open("main-result.txt", "w", encoding="utf-8") as f:
        f.write(result)


def profile(prompt, draft_model, model, max_new_tokens=10000):
    # Initialize the hybrid generator once
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=model,
        device="cuda",
        dtype=torch.float16,
    )

    profile = generator.generate_with_profile(
        prompt=input,
        max_new_tokens=max_new_tokens,
    )

    profile.print_summary()

    # # 分析分布
    analysis = profile.analyze_distribution("uncertainty", percentiles=[10, 20, 30])
    print(f"Top 10% 的 token 不确定性 >= {analysis['percentiles']['top_10%']:.4f}")

    # # 获取最不确定的 tokens
    # high_uncertain = profile.get_high_uncertainty_tokens(10)
    # for token in high_uncertain:
    #     print(
    #         f"位置 {token.position}: '{token.token_text}' "
    #         f"(不确定性: {token.aleatoric_uncertainty:.4f})"
    #     )

    # # 可视化分布 (需要 matplotlib)
    profile.plot_distributions("profile.png")

    # # 导出数据
    profile.save_to_file("profile.json")

    # # 生成 token 序列的火焰图可视化 (基于 entropy)
    profile.visualize_token_sequence(
        save_path="token_sequence_entropy.pdf",
        metric="entropy",  # 或使用 "uncertainty"
        tokens_per_line=15,  # 每行显示的 token 数量
    )


def simple_generate(prompt, draft_model, max_new_tokens=1000):
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=None,
        device="cuda",
        dtype=torch.float16,
    )
    result, stats = generator.simple_generate_with_slm(
        prompt=input,
        max_new_tokens=max_new_tokens,
    )
    print(f"Statatics: {stats}")


if __name__ == "__main__":
    run_hybrid_generation(
        input,
        slm,
        llm,
        "entropy",
        # "semantic_enhanced_route",
        threshold=0.2,
        max_new_tokens=1000,
        verbose=False,
    )
    # profile(input, draft_model, None, 2000)
    # simple_generate(input, draft_model, 2000)
    print("OK")
