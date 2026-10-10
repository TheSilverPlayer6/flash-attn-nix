#!/usr/bin/env python3
"""Functional smoke test for the nix-built flash-attn package.

Requires a CUDA GPU whose SM architecture is among the ones the package was
built for (FLASH_ATTN_CUDA_ARCHS / cudaArchs). Checks forward and backward
outputs of flash_attn_func against a float32 reference implementation.

Usage:
    nix shell <flake>#flash-attn --command python tests/smoke-test.py
"""

import math

import torch
from flash_attn import flash_attn_func

import flash_attn

print("flash_attn:", flash_attn.__version__)
print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA GPU visible (or driver too old for this CUDA major version)"
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"GPU: {name} (sm_{major}{minor})")

torch.manual_seed(0)

B, S, H, D = 2, 1024, 8, 64
dev = "cuda"


def reference(q, k, v, causal):
    """Plain float32 attention on (B, S, H, D) tensors -> (B, S, H, D)."""
    qq, kk, vv = (t.transpose(1, 2).float() for t in (q, k, v))  # (B, H, S, D)
    attn = qq @ kk.transpose(-1, -2) / math.sqrt(D)
    if causal:
        mask = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    out = torch.softmax(attn, dim=-1) @ vv
    return out.transpose(1, 2)  # back to (B, S, H, D), still float32


def run(dtype, fwd_tol, bwd_tol):
    print(f"--- dtype={dtype} ---")
    q = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device=dev, dtype=dtype, requires_grad=True)

    for causal in (False, True):
        out = flash_attn_func(q, k, v, causal=causal)
        assert out.shape == q.shape, (out.shape, q.shape)
        assert out.dtype == dtype

        ref = reference(q, k, v, causal)
        fwd_err = (out.float() - ref).abs().max().item()
        print(f"causal={causal!s:5} | fwd max abs err: {fwd_err:.5f}")
        assert fwd_err < fwd_tol, fwd_err

        g = torch.randn_like(out)
        out.backward(g)

        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        reference(qr, kr, vr, causal).backward(g)

        for label, got, want in (("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)):
            err = (got.float() - want.float()).abs().max().item()
            print(f"causal={causal!s:5} | {label} max abs err: {err:.5f}")
            assert err < bwd_tol, (label, err)

        q.grad = k.grad = v.grad = None


run(torch.float16, fwd_tol=2e-2, bwd_tol=5e-2)
run(torch.bfloat16, fwd_tol=8e-2, bwd_tol=2e-1)

print("SMOKE TEST OK")
