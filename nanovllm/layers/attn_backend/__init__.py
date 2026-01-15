from nanovllm.layers.attn_backend.base import AttnBackend
from nanovllm.layers.attn_backend.flash_attn import FlashAttnBackend
from nanovllm.layers.attn_backend.naive import NaiveAttnBackend

__all__ = ["AttnBackend", "FlashAttnBackend", "NaiveAttnBackend", "get_attn_backend"]


def get_attn_backend(name: str = "flash_attn", *args, **kwargs) -> AttnBackend:
    backend_name = name.lower()
    if backend_name in ("flash_attn", "flash", "flashattention"):
        backend_cls = FlashAttnBackend
    elif backend_name in ("naive", "pytorch"):
        backend_cls = NaiveAttnBackend
    else:
        raise ValueError(f"Unknown attention backend: {name}")
    return backend_cls(*args, **kwargs)
