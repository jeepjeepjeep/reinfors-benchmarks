#!/usr/bin/env bash
# Scaffold: build OpenSpiel from source with the libtorch AlphaZero example (the C++ "all native"
# path the pip wheel does not ship). Writes everything into open_spiel_cpp/ (gitignored).
#
# NOTE (macOS arm64): upstream's libtorch AZ is CPU/CUDA oriented; expect CPU-only here. The build
# downloads libtorch (~large) and takes a while. Run manually, read errors, iterate.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p open_spiel_cpp
cd open_spiel_cpp

if [ ! -d open_spiel ]; then
  git clone https://github.com/google-deepmind/open_spiel.git
fi
cd open_spiel

# install.sh fetches dependencies (abseil, etc.)
./install.sh

export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
# see open_spiel/scripts/global_variables.sh for the libtorch URL it will download; on macOS arm64
# you may need to point OPEN_SPIEL_LIBTORCH_URL at the CPU arm64 libtorch build.

mkdir -p build && cd build
cmake -DPython3_EXECUTABLE="$(command -v python3)" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" ../open_spiel
make -j"$(sysctl -n hw.ncpu)" alpha_zero_torch_example
echo "binary: $(find . -name 'alpha_zero_torch_example' -type f)"
