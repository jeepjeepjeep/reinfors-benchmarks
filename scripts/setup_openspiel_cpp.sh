#!/usr/bin/env bash
# Build OpenSpiel from source with the libtorch AlphaZero example (the C++ "all native" path the
# pip wheel does not ship). Writes everything into open_spiel_cpp/ (gitignored).
#
# Notes:
#  - libtorch AZ needs LIBTORCH + LIBNOP enabled BEFORE install.sh (that step downloads them).
#  - default libtorch URL upstream is Linux+CUDA; overridden here to the macOS arm64 CPU build
#    matching upstream's pinned 2.3.0. There is no MPS/Metal libtorch AZ upstream — CPU-only on
#    macOS, which is itself a benchmark datapoint.
#  - Python bindings are disabled: we only need the C++ example, and it sidesteps the known
#    libtorch/pybind11 interference (open_spiel issue #966).
set -euo pipefail
cd "$(dirname "$0")/.."

export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON
export OPEN_SPIEL_BUILD_WITH_PYTHON=OFF
export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip"

mkdir -p open_spiel_cpp
cd open_spiel_cpp

# pinned: master's libtorch build is broken since 86fe553c (deleted libnop/libtorch CMakeLists
# but left the root references); d15d49f8 is its parent — the last commit where it builds.
OPEN_SPIEL_COMMIT=d15d49f8

if [ ! -d open_spiel ]; then
  git clone https://github.com/google-deepmind/open_spiel.git
fi
cd open_spiel
git checkout -q "$OPEN_SPIEL_COMMIT"

# fetches abseil/json/... plus (with the flags above) libtorch and libnop; cached + resumable
./install.sh

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  ../open_spiel
make -j"$(sysctl -n hw.ncpu)" alpha_zero_torch_example
echo "binary: $(find . -name 'alpha_zero_torch_example' -type f)"
