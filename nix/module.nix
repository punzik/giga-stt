{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.giga-stt;
in
{
  options.services.giga-stt = {
    enable = lib.mkEnableOption "GigaAM speech-to-text REST service";

    package = lib.mkPackageOption pkgs "giga-stt" { };

    # A directory with the model weights (*.ckpt / *_tokenizer.model) from the nix store.
    # null = the weights are downloaded into the StateDirectory on first start (internet required).
    modelFiles = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      description = "Prefetched model files (packages.gigaam-model)";
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "v3_e2e_rnnt";
      description = "Russian GigaAM model name";
    };

    graniteModel = lib.mkOption {
      type = lib.types.str;
      default = "ibm-granite/granite-speech-5.0-470m-turboctc";
      description = "English IBM Granite Speech TurboCTC model name";
    };

    device = lib.mkOption {
      type = lib.types.str;
      default = "auto";
      description = "auto / cuda / cpu";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
    };

    vadBackend = lib.mkOption {
      type = lib.types.enum [
        "auto"
        "pyannote"
        "silero"
        "energy"
      ];
      default = "auto";
      description = "Segmentation backend for audio longer than 25 s";
    };

    frBatchSize = lib.mkOption {
      type = lib.types.int;
      default = 4;
      description = "Chunk batch size (increase for GPUs with more VRAM)";
    };

    diarization = lib.mkOption {
      type = lib.types.enum [ "off" "pyannote" ];
      default = "pyannote";
      description = ''
        Speaker diarization backend for with_speakers=true requests.
        pyannote requires an HF_TOKEN with accepted terms for the gated
        model pyannote/speaker-diarization-community-1 (via environmentFile
        or extraEnv).
      '';
    };

    maxUploadMb = lib.mkOption {
      type = lib.types.int;
      default = 100;
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "systemd EnvironmentFile (GIGAAM_API_KEY, HF_TOKEN, …)";
    };

    extraEnv = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };
  };

  config = lib.mkIf cfg.enable {
    nixpkgs.overlays = [
      (final: prev: {
        giga-stt = final.callPackage ./package.nix { };
      })
    ];

    systemd.services.giga-stt = {
      description = "GigaAM and IBM Granite speech-to-text REST service";
      documentation = [ "https://github.com/salute-developers/GigaAM" ];
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      preStart = lib.mkIf (cfg.modelFiles != null) (
        lib.getExe (
          pkgs.writeShellScript "giga-stt-prestart" ''
            mkdir -p /var/lib/giga-stt/models
            for f in ${cfg.modelFiles}/*; do
              [ -e "/var/lib/giga-stt/models/$(basename "$f")" ] || ln -s "$f" /var/lib/giga-stt/models/
            done
          ''
        )
      );

      environment = {
        GIGAAM_MODEL = cfg.model;
        GIGAAM_DEVICE = cfg.device;
        GIGAAM_HOST = cfg.host;
        GIGAAM_PORT = toString cfg.port;
        GIGAAM_CACHE_DIR = "/var/lib/giga-stt/models";
        GRANITE_MODEL = cfg.graniteModel;
        GRANITE_CACHE_DIR = "/var/lib/giga-stt/models/huggingface";
        GIGAAM_VAD_BACKEND = cfg.vadBackend;
        GIGAAM_FR_BATCH_SIZE = toString cfg.frBatchSize;
        GIGAAM_DIARIZATION = cfg.diarization;
        PYANNOTE_METRICS_ENABLED = "0";
        GIGAAM_MAX_UPLOAD_MB = toString cfg.maxUploadMb;
        GIGAAM_MAX_AUDIO_SEC = "7200";
      } // cfg.extraEnv;

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/giga-stt";
        WorkingDirectory = "/var/lib/giga-stt";
        StateDirectory = "giga-stt";
        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        PrivateTmp = true;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}