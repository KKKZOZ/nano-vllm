import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=20480,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1000)
    prompts = [
        "Let $O(0,0), A(\\tfrac{1}{2}, 0),$ and $B(0, \\tfrac{\\sqrt{3}}{2})$ be points in the coordinate plane. Let $\\mathcal{F}$ be the family of segments $\\overline{PQ}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\\overline{AB}$, distinct from $A$ and $B$, that does not belong to any segment from $\\mathcal{F}$ other than $\\overline{AB}$. Then $OC^2 = \\tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p + q$."
        * 1,
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        # print(f"Token ids: {output['token_ids']}")


if __name__ == "__main__":
    main()
