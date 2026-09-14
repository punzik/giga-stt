{
  description = "giga-stt: Russian and English REST speech-to-text service";

  inputs = {
    # The nixpkgs revision matches the system channel (NixOS 26.05) —
    # maximum overlap with the local store and the cache.nixos.org cache.
    nixpkgs.url = "github:NixOS/nixpkgs/04607e1165ac";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        # NVIDIA wheels from PyPI — non-free license
        config.allowUnfree = true;
      };
    in
    {
      packages.x86_64-linux = rec {
        giga-stt = pkgs.callPackage ./package.nix { };
        gigaam-model = giga-stt.passthru.gigaamModel;
        default = giga-stt;
      };

      apps.x86_64-linux = rec {
        giga-stt = {
          type = "app";
          program = "${self.packages.x86_64-linux.giga-stt}/bin/giga-stt";
        };
        default = giga-stt;
      };

      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [ self.packages.x86_64-linux.giga-stt ];
      };

      overlays.default = final: prev: {
        giga-stt = final.callPackage ./package.nix { };
      };

      # The module lives in nix/default.nix, so `imports = [ <path>/nix ]`
      # (directory) plugs the service module straight into a NixOS config.
      nixosModules.giga-stt = ./default.nix;
      nixosModules.default = self.nixosModules.giga-stt;
    };
}