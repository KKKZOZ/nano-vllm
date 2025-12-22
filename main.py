import torch
from transformers import AutoTokenizer

# from generate import (
#     simple_generate,
# )
from hybrid_generator import HybridGenerator

# prompt = "Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$."
input = "write a simple calculator in python"
model = "/root/huggingface/Qwen3-8B"
draft_model = "/root/huggingface/Qwen3-0.6B"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": input},
]

tokenizer = AutoTokenizer.from_pretrained(model)
input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# result = simple_generate("/root/huggingface/Qwen3-0.6B", input, max_new_tokens=1000)

# result = simple_generate("/root/huggingface/Qwen3-1.7B", input, max_new_tokens=1000)

# result = simple_generate("/root/huggingface/Qwen3-8B", input, max_new_tokens=1000)

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
    prompt, draft_model, model, threshold=0.1, max_new_tokens=1000, verbose=False
):
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=model,
        device="cuda",
        dtype=torch.float16,
        verbose=verbose,
    )
    result, stats = generator.generate(
        prompt=input,
        strategy="entropy",
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        threshold=threshold,
    )
    print(f"Statatics: {stats}")


def profile(prompt, draft_model, model):
    # Initialize the hybrid generator once
    generator = HybridGenerator(
        slm_model_id=draft_model,
        llm_model_id=model,
        device="cuda",
        dtype=torch.float16,
    )

    profile = generator.generate_with_profile(
        prompt=input,
        max_new_tokens=20000,
        threshold=0.1,
        routing_metric="entropy",
        enable_routing=False,
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


if __name__ == "__main__":
    run_hybrid_generation(
        input, draft_model, model, threshold=0.5, max_new_tokens=1000, verbose=False
    )
    print("OK")
