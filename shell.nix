{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python314
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

shellHook = ''
  export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

  [ -d .venv ] || ./setup.sh
  source .venv/bin/activate
'';
}
