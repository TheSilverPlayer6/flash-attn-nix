# flash-attn-nix

[`flash-attn`](https://github.com/Dao-AILab/flash-attention) (FlashAttention-2)
packaged for [Nix](https://nixos.org/) with `buildPythonPackage` — a fully
hermetic, from-source CUDA build. No prebuilt wheels are downloaded at build
time; the CUDA kernels are compiled with nixpkgs' `nvcc` inside the Nix build
sandbox.

- **Package:** `flash-attn` **2.8.3.post1** (latest upstream stable release)
- **Python:** nixpkgs default (`python3`)
- **PyTorch:** nixpkgs `torch-bin` — the official PyTorch **cu130** manylinux
  wheel, repackaged against nixpkgs' CUDA 13 redistributables (overrideable)
- **CUDA:** nixpkgs `cudaPackages_13_0` (nvcc **13.0.3**, via
  `cudaPackages.backendStdenv` with a compatible gcc)
- **Platform:** `x86_64-linux`

## Quick start

```console
# build it (see the build-cost warning below first!)
nix build github:TheSilverPlayer6/flash-attn-nix

# or get a shell with python + flash_attn + torch (CUDA)
nix develop github:TheSilverPlayer6/flash-attn-nix
python -c 'import flash_attn; print(flash_attn.__version__)'

# or install into your profile
nix profile install github:TheSilverPlayer6/flash-attn-nix
```

> [!WARNING]
> **This package compiles ~100 heavy CUDA translation units.** With the default
> architecture list (`80;90;100;120`) expect roughly **1–3 h** on a 16-core /
> 64 GiB machine (plus ~30 min for `torch-bin` if it is not in your store yet).
> Upstream's setup.py caps `MAX_JOBS` at ~1 job per 9 GiB of *available* RAM —
> each nvcc job peaks around 8–9 GiB with the default `NVCC_THREADS=4`.
> If you only care about one GPU family, restrict `cudaArchs` (below).

As a flake input:

```nix
{
  inputs.flash-attn-nix.url = "github:TheSilverPlayer6/flash-attn-nix";

  outputs = { nixpkgs, flash-attn-nix, ... }: {
    # the overlay adds python3Packages.flash-attn / pkgs.flash-attn / pkgs.mkFlashAttn
    nixosConfigurations.foo = nixpkgs.lib.nixosSystem {
      modules = [ { nixpkgs.overlays = [ flash-attn-nix.overlays.default ]; } ];
    };
  };
}
```

Note: the overlay needs `nixpkgs.config.allowUnfree = true` in your
configuration (the official torch wheel, CUDA redistributables and cuDNN are
unfree). This flake sets it for its own `packages`/`devShells` outputs.

## Customizing

`pkgs.mkFlashAttn` (added by the overlay) re-invokes `package.nix` with your
overrides; the derivation itself also supports `.override`:

```nix
# H100/H200 only — much faster build
pkgs.mkFlashAttn { cudaArchs = [ "90" ]; }

# RTX 40/50-series only
pkgs.mkFlashAttn { cudaArchs = [ "80" "90" "120" ]; }

# limit nvcc parallelism / internal threads
pkgs.mkFlashAttn { maxJobs = 4; nvccThreads = 2; }

# build against a from-source CUDA PyTorch instead of torch-bin
# (needs nixpkgs.config.cudaSupport = true; hours of extra build time)
pkgs.mkFlashAttn {
  torch = pkgs.python3Packages.torchWithCuda;
  cudaPackages = pkgs.cudaPackages; # match the CUDA version torch was built with
}
```

`cudaArchs` maps to upstream's `FLASH_ATTN_CUDA_ARCHS`; supported values are
`80`, `90`, `100`, `120` (gated by nvcc version in upstream's setup.py).
Set `cudaArchs = null` to keep upstream's default.

## How it works / design notes

`package.nix` is a single `buildPythonPackage` call (over
`stdenv = cudaPackages.backendStdenv` so nvcc gets a supported host gcc — the
same pattern nixpkgs uses for its CUDA torch build and `tiny-cuda-nn`):

- **Source**: `fetchFromGitHub` at tag `v2.8.3.post1`; the `csrc/cutlass`
  submodule is fetched separately at its pinned commit
  (`dc48179`) and symlinked into place in `postPatch`. The ROCm-only
  `csrc/composable_kernel` submodule is *not* fetched.
- **Build system**: upstream ships a legacy `setup.py` (no `pyproject.toml`),
  so the wheel is produced with `build-system = [ setuptools wheel ]` through
  nixpkgs' standard PEP-517 hooks (`pyproject-build --no-isolation`).
- **Patch** (`patches/setup-py-nix-fixes.patch`):
  1. `from wheel.bdist_wheel import bdist_wheel` fails with nixpkgs'
     wheel ≥ 0.46 → fall back to `setuptools.command.bdist_wheel`.
  2. `setup_requires=[packaging, psutil, ninja]` would make setuptools try to
     download from PyPI (impossible: the build sandbox has no network) →
     emptied; everything is provided via `nativeBuildInputs` instead.
- **Hermeticity**: `FLASH_ATTENTION_FORCE_BUILD=TRUE` disables upstream's
  "guess a prebuilt wheel URL and download it" logic in `bdist_wheel`.
- **`CUDA_HOME`**: a `symlinkJoin` of `cuda_nvcc`, `cuda_cudart` (dev+lib,
  incl. `lib/stubs/libcuda.so` for `LIBRARY_PATH`), `cccl`, `cuda_nvrtc` and
  `cuda_nvtx`, with a `lib64 → lib` alias, matching what torch's
  `cpp_extension` expects.
- **torch** is both a `nativeBuildInput` (setup.py imports it to read
  `torch.__version__`, the C++11 ABI flag and the extension's include/lib
  paths) and the propagated runtime dependency, guaranteeing the kernels are
  compiled against exactly the torch they will run with.
- **Runtime deps**: `dependencies = [ einops torch ]` satisfy the wheel
  METADATA (`Requires-Dist: torch, einops`) via nixpkgs' runtime-deps check.
- **Checks**: `pythonImportsCheck = [ "flash_attn" ]` runs in the sandbox
  (importing the extension needs no GPU/driver — CUDA initializes lazily and
  `libcuda.so.1` is dlopen'd only for actual GPU use). Upstream's pytest suite
  needs a physical GPU, so `doCheck = false`; run `tests/` on a GPU machine:

  ```console
  nix shell .#flash-attn --command python tests/import-test.py
  nix shell .#flash-attn --command python tests/smoke-test.py   # needs GPU
  ```

### Runtime requirements

- An NVIDIA driver supporting **CUDA 13.0** (≥ 580.xx). The nix store paths
  carry everything else (cudart, cublas, cudnn, … from `cudaPackages_13_0` /
  the torch wheel); only the kernel-mode driver + `libcuda.so.1` come from the
  host. On NixOS use `hardware.graphics.enable` + the NVIDIA driver as usual;
  elsewhere make sure `libcuda.so.1` is discoverable (e.g. via
  `LD_LIBRARY_PATH` in containers).
- A GPU with an SM architecture included in `cudaArchs` (default: sm_80
  Ampere, sm_90 Hopper, sm_100/sm_120 Blackwell).

## CI

GitHub Actions evaluates the flake, the derivation and the full build graph
(`nix flake check --no-build`, `nix eval`, `nix build --dry-run`). Compiling
the kernels is far beyond hosted-runner capacity/timeouts, so the full build
is verified manually — see below.

## Verification

- [x] `nix flake check` passes
- [x] `nix build` completes end-to-end from a fresh clone (all default
      architectures `80;90;100;120`)
- [x] `tests/import-test.py` passes (no GPU needed)
- [x] `tests/smoke-test.py` passes on a real GPU (forward + backward vs.
      float32 reference, fp16 + bf16)

## License

- `flash-attn` itself: BSD-3-Clause (Tri Dao et al.)
- This packaging: BSD-3-Clause (see header of `package.nix` — pick your
  favorite permissive terms; no warranty).
- Beware: building with the default wiring pulls in unfree redistributables
  (official PyTorch wheel, CUDA redistributables, cuDNN) — hence
  `allowUnfree`.
