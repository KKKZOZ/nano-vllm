# NOTE

## KV Shape

nano-vllm 使用的 KV 格式是 

```python
self.kv_cache = torch.empty(
    2,
    hf_config.num_hidden_layers,
    config.num_kvcache_blocks,
    self.block_size,
    num_kv_heads,
    head_dim,
)

```

也就是 NHD

后三维没什么问题，但 flash-infer 期望的两种具体的 kv cache 格式：


```python
# 5-D tensor
kv_cache_nhd = torch.empty(max_num_pages, 2, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout

# 4-D tensor
# kv_data = (k_data, v_data)
k_cache_nhd = torch.empty(max_num_pages, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout

v_cache_nhd = torch.empty(max_num_pages, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout
```

我们使用 4-D tensor 的格式会好一点

现在 nano-vllm 中的格式正好符合

## flash-infer call

首先替换 attention.py 中的 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache`

+ `flash_attn_varlen_func` -> `BatchPrefillWithPagedKVCacheWrapper`
+ `flash_attn_with_kvcache` -> `BatchDecodeWithPagedKVCacheWrapper`

好像其他都不用改？
