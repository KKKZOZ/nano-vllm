from nanovllm import Backend

if __name__ == "__main__":
    model = "/root/huggingface/Qwen3-8B"
    backend = Backend(model, max_num_seqs=1, enable_extend_cudagraph=True)

    seq_id = 0
    prefill_tokens = list(range(1, 257))  # Use token IDs
    print(f"running seq_id={seq_id} prefill...")
    logits = backend.forward(seq_id, prefill_tokens)
    print(f"Logits: {logits[-1, :100].tolist()}")
    seq_id += 1
    print(f"running seq_id={seq_id} prefill...")
    logits = backend.forward(seq_id, prefill_tokens)
    print(f"Logits: {logits[-1, :100].tolist()}")
