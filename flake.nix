{
  description = "flash-attn (FlashAttention-2) packaged with Nix's buildPythonPackage — hermetic CUDA build";

  inputs = {
    # nixos-unstable as of 2026-10-08
    nixpkgs.url = "github:NixOS/nixpkgs/e7439b6b14ad3cc35d05608ebca9bce01a25f5f8";
  };

  outputs =
    { self, nixpkgs }:
    let
      # The default wiring builds against nixpkgs' `torch-bin` — the official
      # PyTorch cu130 manylinux wheel, repackaged and linked against nixpkgs'
      # CUDA 13 redistributables. That means:
      #   * no multi-hour PyTorch source build,
      #   * the same situation upstream's own `pip install flash-attn` wheels
      #     are built for (official torch wheel + system CUDA),
      #   * CUDA 13.0 nvcc from nixpkgs compiling the kernels.
      # Everything is overridable; see `mkFlashAttn` below and the README.
      system = "x86_64-linux";

      overlay = final: prev:
        let
          # Matches torch-bin's cu130 wheels; CUDA 13.0.3 redistributables.
          cudaPackages = final.cudaPackages_13_0;
          torch = final.python3Packages.torch-bin.override { inherit cudaPackages; };
        in
        {
          python3 = prev.python3.override {
            packageOverrides = pfinal: pprev: {
              flash-attn = pfinal.callPackage ./package.nix {
                inherit cudaPackages torch;
              };
            };
          };
          python3Packages = final.python3.pkgs;

          # Convenience alias.
          flash-attn = final.python3Packages.flash-attn;

          # Builder with this flake's defaults, e.g. in your own overlay:
          #   mkFlashAttn { cudaArchs = [ "90" ]; }          # H100-only, much faster build
          #   mkFlashAttn { torch = myTorch; cudaPackages = pkgs.cudaPackages_12_9; }
          mkFlashAttn =
            args:
            final.python3Packages.callPackage ./package.nix (
              {
                inherit cudaPackages torch;
              }
              // args
            );
        };

      pkgs = import nixpkgs {
        inherit system;
        # CUDA redistributables, cuDNN and the official torch wheel are unfree.
        config.allowUnfree = true;
        overlays = [ overlay ];
      };
    in
    {
      overlays.default = overlay;

      packages.${system} = {
        default = pkgs.flash-attn;
        flash-attn = pkgs.flash-attn;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (pkgs.python3.withPackages (ps: [ ps.flash-attn ]))
        ];
      };

      # Forcing `.drvPath` here makes `nix flake check` instantiate the whole
      # derivation graph (every input, hash and argument of the package is
      # fully evaluated). The string context is discarded so that the .drv
      # path does not become a store reference of this check derivation.
      checks.${system}.evaluate = pkgs.runCommand "flash-attn-eval-check" { } ''
        echo ${builtins.unsafeDiscardStringContext pkgs.flash-attn.drvPath} > $out
      '';
    };
}
