{
  lib,
  stdenv,
  python3Packages,
  fetchFromGitHub,
  fetchurl,
  runCommand,
  ffmpeg,
  fetchgit
}:
let
  # PyPI wheels of the CUDA stack (torch/nvidia/triton/silero-vad) — see wheels.nix
  wheels = import ./wheels.nix {
    inherit lib;
    inherit fetchurl;
    inherit python3Packages;
  };

  # The gigaam package, pinned to a working upstream commit.
  # The dependency pins in the upstream pyproject (onnx==1.19.* etc.) are ignored by nix:
  # versions come from nixpkgs and do not affect onnx/onnxruntime at service runtime.
  gigaam = python3Packages.buildPythonPackage {
    pname = "gigaam";
    version = "0.2.0";
    pyproject = true;
    build-system = with python3Packages; [ setuptools ];
    src = fetchFromGitHub {
      owner = "salute-developers";
      repo = "GigaAM";
      rev = "7447938d791c4f3e643386ee22c33777004293a5";
      hash = "sha256-Lo3q8nhMkttgUCWRUv4ChGYcZlcsn/yNEHW020stPNw=";
    };
    propagatedBuildInputs = with python3Packages; [
      hydra-core
      numpy
      omegaconf
      onnx
      onnxruntime
      soundfile
      sentencepiece
      tqdm
    ];
    # upstream pins (onnx==1.19.* etc.) do not match nixpkgs versions
    dontCheckRuntimeDeps = true;
    doCheck = false;
  };

  # Speaker-diarization stack (pyannote/speaker-diarization-community-1).
  # nixpkgs' pyannote-audio is overridden to use the same torch/torchaudio
  # wheels as the rest of the package — otherwise the derivation would pull in
  # a second, incompatible torch from nixpkgs.
  pyannoteAudio = python3Packages.pyannote-audio.override {
    inherit (wheels) torch torchaudio torchcodec;
  };

  # The GigaAM-v3-e2e-rnnt checkpoint and tokenizer from the official Sber CDN.
  # The md5 check inside gigaam.load_model confirms download integrity.
  gigaamModel = runCommand "gigaam-v3_e2e_rnnt-model" { } ''
    mkdir -p $out
    ln -s ${
      fetchurl {
        url = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt.ckpt";
        hash = "sha256-9gxi/kWQLZZwAHcKEvJg3nxaxNS45ehS8d+MVI45WLU=";
      }
    } $out/v3_e2e_rnnt.ckpt
    ln -s ${
      fetchurl {
        url = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt_tokenizer.model";
        hash = "sha256-gowSyZEBnu+VKpYGYfJaktatJ5WR4upGa0rt3x0goYo=";
      }
    } $out/v3_e2e_rnnt_tokenizer.model
  '';
in
python3Packages.buildPythonApplication {
  pname = "giga-stt";
  version = "1.0.0";
  pyproject = true;
  # To package a released source, replace the local path below with this block:
  # src = fetchgit {
  #   url = "https://git.embddr.xyz/np/giga-stt.git";
  #   rev = "<GIGA_STT_REV>";
  #   hash = "sha256-<GIGA_STT_SOURCE_HASH>";
  # };
  src = ../.;

  build-system = with python3Packages; [ setuptools ];

  propagatedBuildInputs =
    (with python3Packages; [
      fastapi
      uvicorn
      python-multipart
      soundfile
      numpy
      hydra-core
      omegaconf
      onnx
      onnxruntime
      sentencepiece
      tqdm
      # runtime dependencies of the torch wheel (filelock, sympy, etc.)
      filelock
      typing-extensions
      sympy
      networkx
      jinja2
      fsspec
      # Runtime dependencies of the Transformers wheel.
      huggingface-hub
      packaging
      pyyaml
      regex
      requests
      # Lets huggingface_hub use a SOCKS proxy when one is configured.
      socksio
    ])
    ++ [
      gigaam
      wheels.torch
      wheels.torchaudio
      wheels.nvidia_cublas_cu12
      wheels.nvidia_cuda_cupti_cu12
      wheels.nvidia_cuda_nvrtc_cu12
      wheels.nvidia_cuda_runtime_cu12
      wheels.nvidia_cudnn_cu12
      wheels.nvidia_cufft_cu12
      wheels.nvidia_cufile_cu12
      wheels.nvidia_curand_cu12
      wheels.nvidia_cusolver_cu12
      wheels.nvidia_cusparse_cu12
      wheels.nvidia_cusparselt_cu12
      wheels.nvidia_nccl_cu12
      wheels.nvidia_nvjitlink_cu12
      wheels.nvidia_nvtx_cu12
      wheels.triton
      wheels.silero_vad
      wheels.transformers
      wheels.tokenizers
      wheels.safetensors
      pyannoteAudio
    ];

  # Wrapper: ffmpeg in PATH (needed by gigaam.load_audio and ffprobe);
  # LD_LIBRARY_PATH — libstdc++ for the manylinux wheels and the CUDA driver on NixOS.
  postFixup = ''
    wrapProgram $out/bin/giga-stt \
      --prefix PATH : ${lib.makeBinPath [ ffmpeg ]} \
      --prefix LD_LIBRARY_PATH : ${lib.getLib stdenv.cc.cc}/lib \
      --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib
  '';

  passthru = {
    inherit gigaamModel;
    meta = {
      description = "Russian and English REST speech-to-text service (GigaAM and IBM Granite)";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  };
}
