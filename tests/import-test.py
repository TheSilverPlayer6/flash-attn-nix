#!/usr/bin/env python3
"""Import check for the nix-built flash-attn package.

Works without a GPU: importing flash_attn loads the compiled
flash_attn_2_cuda extension (torch + libcudart must resolve), but CUDA
itself is only initialized lazily.
"""

import flash_attn
import flash_attn_2_cuda  # the compiled CUDA extension
import torch

print("flash_attn:", flash_attn.__version__)
print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
print("extension:", flash_attn_2_cuda.__file__)
print("cuda available:", torch.cuda.is_available())

for fn in (
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_with_kvcache",
):
    assert hasattr(flash_attn, fn), f"missing {fn}"

print("IMPORT TEST OK")
