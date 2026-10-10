{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  symlinkJoin,

  # CUDA package set providing both the nvcc toolchain and the libraries the
  # extension links against. Must match the CUDA major version that the chosen
  # `torch` build was compiled against (flake.nix wires up CUDA 13.0 to match
  # torch-bin's official cu130 wheels).
  cudaPackages,

  # PEP-517 build system. Upstream still ships a legacy setup.py without a
  # pyproject.toml, so the wheel is produced via the setuptools backend.
  setuptools,
  wheel,

  # setup.py imports these at build time
  ninja,
  packaging,
  psutil,

  # runtime dependencies (setup.py: install_requires)
  einops,

  # A CUDA-enabled torch. setup.py imports it at build time to read
  # torch.__version__, the C++11 ABI flag and the include/library paths the
  # extension is compiled and linked against. Must be the same torch the
  # package will be used with at runtime.
  torch,

  # --- knobs ----------------------------------------------------------------
  #
  # FLASH_ATTN_CUDA_ARCHS: CUDA SM architectures to compile kernels for.
  # Upstream's default is "80;90;100;120" (Ampere, Hopper, Blackwell
  # datacenter, Blackwell GeForce). Restricting this (e.g. [ "90" ] when
  # targeting H100/H200 only) drastically cuts build time and memory usage.
  # Set to null to keep upstream's default.
  cudaArchs ? [
    "80"
    "90"
    "100"
    "120"
  ],

  # MAX_JOBS: number of parallel nvcc invocations driven by ninja.
  # null => let setup.py's psutil heuristic decide:
  #         min(cores / 2, available RAM [GiB] / 9).
  maxJobs ? null,

  # NVCC_THREADS: internal nvcc parallelism per translation unit
  # (upstream default: 4).
  nvccThreads ? null,
}:

let
  version = "2.8.3.post1";

  # CUDA components flash-attn's setup.py / torch's cpp_extension expect to
  # find under $CUDA_HOME (bin/nvcc, include/, lib{,64}/, lib/stubs, nvvm/).
  cuda-libs = with cudaPackages; [
    (lib.getDev cuda_cudart) # cuda_runtime.h, cuda.h
    (lib.getLib cuda_cudart) # libcudart.so, lib/stubs/libcuda.so
    (lib.getDev cccl) # <cub/...>, <thrust/...>, <nv/target>
    (lib.getInclude cuda_nvrtc) # nvrtc.h
    (lib.getLib cuda_nvrtc) # libnvrtc.so
    (lib.getInclude cuda_nvtx) # nvToolsExt.h / nvtx3
    (lib.getLib cuda_nvtx) # libnvToolsExt.so
  ];

  # $CUDA_HOME for build-time tools: nvcc plus headers and libraries.
  cuda-home-native = symlinkJoin {
    name = "cuda-home-native";
    paths = [ cudaPackages.cuda_nvcc ] ++ cuda-libs;
    postBuild = ''
      ln -sfn "$out/lib" "$out/lib64"
    '';
  };

  # Headers/libraries for the compile and link steps (host-side).
  cuda-home = symlinkJoin {
    name = "cuda-home";
    paths = cuda-libs;
    postBuild = ''
      ln -sfn "$out/lib" "$out/lib64"
    '';
  };

  # The csrc/cutlass git submodule pin of flash-attention v${version}.
  # Fetched separately (instead of fetchSubmodules = true) to avoid pulling
  # in the huge csrc/composable_kernel submodule, which is ROCm-only.
  cutlass = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "dc4817921edda44a549197ff3a9dcf5df0636e7b";
    hash = "sha256-I//kXsZoOGD2ru/Iln2xHQ12EE2jz0zo2mA5YyWyjRg=";
  };
in
assert lib.assertMsg (torch.cudaSupport or false) ''
  flash-attn requires a CUDA-enabled torch. Pass one explicitly, e.g.:

    torch = pkgs.python3Packages.torch-bin.override {
      cudaPackages = pkgs.cudaPackages_13_0;
    };

  or, for a from-source PyTorch: pkgs.python3Packages.torchWithCuda
  (built with `nixpkgs.config.cudaSupport = true`).
'';
# nvcc only supports a limited set of host compilers; cudaPackages.backendStdenv
# selects a compatible gcc stdenv (this is the same approach nixpkgs itself uses
# for the CUDA build of torch and for tiny-cuda-nn).
(buildPythonPackage.override { stdenv = cudaPackages.backendStdenv; })
{
  pname = "flash-attn";
  inherit version;

  # Upstream has no pyproject.toml; `pyproject = true` selects nixpkgs'
  # standard PEP-517 hooks, which fall back to the legacy setuptools backend.
  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];

  src = fetchFromGitHub {
    owner = "Dao-AILab";
    repo = "flash-attention";
    tag = "v${version}";
    hash = "sha256-oZTZegr8hNOV5IrkzeU3CpWTSffqDgV+kGvwhpwBkHE=";
  };

  patches = [
    # - import bdist_wheel from setuptools when wheel >= 0.46 removed it
    # - neutralize setup_requires (would attempt PyPI downloads; the Nix
    #   build sandbox has no network access)
    ./patches/setup-py-nix-fixes.patch
  ];

  postPatch = ''
    # GitHub tarballs do not contain git submodules; link in the pinned
    # cutlass checkout so that csrc/cutlass/include/cutlass/cutlass.h exists.
    ln -s ${cutlass} csrc/cutlass
  '';

  env =
    {
      # Where setup.py and torch's cpp_extension look for nvcc & CUDA libs.
      CUDA_HOME = cuda-home-native;
      # In case anything ever passes -lcuda: resolve against the driver stubs
      # (the real libcuda.so.1 comes from the GPU driver at runtime).
      LIBRARY_PATH = "${cuda-home}/lib/stubs";
      # Build from source; never try to download prebuilt wheels from GitHub
      # releases (impossible offline, and they would not match this build).
      FLASH_ATTENTION_FORCE_BUILD = "TRUE";
    }
    // lib.optionalAttrs (cudaArchs != null) {
      FLASH_ATTN_CUDA_ARCHS = lib.concatStringsSep ";" cudaArchs;
    }
    // lib.optionalAttrs (maxJobs != null) {
      MAX_JOBS = toString maxJobs;
    }
    // lib.optionalAttrs (nvccThreads != null) {
      NVCC_THREADS = toString nvccThreads;
    };

  nativeBuildInputs = [
    ninja
    packaging
    psutil
    torch
  ];

  buildInputs = [
    cuda-home
  ];

  # Propagated as runtime deps (satisfies the wheel METADATA: torch, einops).
  dependencies = [
    einops
    torch
  ];

  # Importing flash_attn loads the compiled flash_attn_2_cuda extension.
  # That needs torch + libcudart but no GPU or driver: CUDA is only
  # initialized lazily, and the extension does not link libcuda directly.
  pythonImportsCheck = [
    "flash_attn"
  ];

  # Upstream's pytest suite requires a physical NVIDIA GPU, which the Nix
  # build sandbox does not have. See tests/ for smoke tests to run on a
  # GPU machine.
  doCheck = false;

  passthru = {
    inherit
      cudaPackages
      cutlass
      ;
  };

  meta = {
    description = "FlashAttention: fast and memory-efficient exact attention with IO-awareness (CUDA build)";
    homepage = "https://github.com/Dao-AILab/flash-attention";
    changelog = "https://github.com/Dao-AILab/flash-attention/releases/tag/v${version}";
    license = lib.licenses.bsd3;
    sourceProvenance = [ lib.sourceTypes.fromSource ];
    # Default wiring targets torch-bin's cu130 manylinux wheels (x86_64 only).
    # For aarch64-linux, override torch with a CUDA-enabled source build.
    platforms = [ "x86_64-linux" ];
    maintainers = [ ];
  };
}
