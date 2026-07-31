#!/usr/bin/env bash
# Build OpenSpiel from source with the libtorch AlphaZero example (the C++ "all native" path the
# pip wheel does not ship). Writes everything into open_spiel_cpp/ (gitignored).
#
# Notes:
#  - libtorch AZ needs LIBTORCH + LIBNOP enabled BEFORE install.sh (that step downloads them).
#  - libtorch URL is platform-selected below, pinned to upstream's 2.3.0 generation: macOS
#    arm64 gets the CPU build (no MPS/Metal libtorch AZ upstream — CPU-only there, itself a
#    benchmark datapoint); Linux gets the cxx11-abi CUDA 12.1 build (the GPU bench box).
#  - Python bindings are disabled: we only need the C++ example, and it sidesteps the known
#    libtorch/pybind11 interference (open_spiel issue #966).
set -euo pipefail
cd "$(dirname "$0")/.."

export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON
export OPEN_SPIEL_BUILD_WITH_PYTHON=OFF
if [[ "$(uname)" == "Darwin" ]]; then
  export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip"
else
  export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip"
  # libtorch 2.3 (cu121 generation) links libnvToolsExt, which CUDA toolkits >= 12.9 no longer
  # ship — steer CMake to the newest installed 12.x that still has it. Build-time discovery
  # only; the driver runs libtorch's bundled kernels regardless of the toolkit picked here.
  for d in /usr/local/cuda-12.*; do
    [[ -e "$d/lib64/libnvToolsExt.so" ]] && cuda12="$d"
  done
  if [[ -n "${cuda12:-}" ]]; then
    export CUDAToolkit_ROOT="$cuda12"
    export PATH="$cuda12/bin:$PATH"
    echo "CUDA toolkit for libtorch discovery: $cuda12"
  fi
fi

mkdir -p open_spiel_cpp
cd open_spiel_cpp

# Master snapshot (2026-07-17), which carries both upstream vpnet perf fixes (Oct 2025
# batched tensor staging; Mar 2026 NoGradGuard + batched output extraction, PR #1488) — the
# benchmark target is their code as maintained today. Master cannot build the libtorch path
# as-is: 86fe553c deleted the libtorch/libnop CMake glue while the root CMakeLists still
# add_subdirectory's both (configure error; unreported upstream as of 2026-07). Restored
# below content-identical from that commit's parent — build glue only, zero behavior.
# (Historical: pre-master-era "as shipped" measurements used pin d15d49f8 +
# scripts/fix_vpnet_gpu_staging.patch, kept for the record.)
OPEN_SPIEL_COMMIT=112b7770

if [ ! -d open_spiel ]; then
  git clone https://github.com/google-deepmind/open_spiel.git
fi
cd open_spiel
git fetch -q origin master
# Every tracked-file modification here is a patch this script manages (instrumentation) —
# reset before switching commits so re-runs are idempotent; patches re-apply below.
git reset --hard -q
git checkout -q "$OPEN_SPIEL_COMMIT"
git checkout -q 86fe553c^ -- open_spiel/libtorch open_spiel/libnop

# Dependency caches are version-pinned by the source tree, and install.sh skips existing
# dirs — so a commit switch can strand them (master's abseil 20250814.1 vs a cached
# 20250127.1 broke the build on the MutexLock API change). Refresh on mismatch, and drop
# the build dir with it: its objects were compiled against the old headers.
want_absl=$(grep -oE 'OPEN_SPIEL_ABSL_VERSION:-"[^"]+"' open_spiel/scripts/global_variables.sh | cut -d'"' -f2)
if [[ -d open_spiel/abseil-cpp ]] && ! git -C open_spiel/abseil-cpp describe --tags 2>/dev/null | grep -q "$want_absl"; then
  echo "refreshing stale abseil-cpp cache -> $want_absl"
  # install.sh's cached_clone keys its download cache by DIRECTORY NAME ONLY (ignores the
  # requested tag) and cp -r's the cached copy back — purge that too or the stale version
  # simply reappears.
  rm -rf open_spiel/abseil-cpp download_cache/abseil-cpp build
fi

# benchmark instrumentation: request/cache/forward counters in VPNetEvaluator ([inst] stderr
# lines, parsed by decompose_sequential.py). Idempotent.
if git apply --check ../../scripts/instrument_vpevaluator.patch 2>/dev/null; then
  git apply ../../scripts/instrument_vpevaluator.patch
fi

# fetches abseil/json/... plus (with the flags above) libtorch and libnop; cached + resumable
./install.sh

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  ../open_spiel
# Portable job count: `sysctl -n hw.ncpu` is macOS-only — on Linux it yields EMPTY, and
# bare `make -j` means UNLIMITED jobs (wedged the 32GB bench box via OOM on a full build).
# Build on a SUBSET of cores (cores - 1): the OS keeps a schedulable core, so even a
# memory/link spike degrades to slow instead of taking sshd down with it.
CORES="$( (sysctl -n hw.ncpu || nproc) 2>/dev/null )"
CORES="${CORES:-4}"
JOBS=$(( CORES > 1 ? CORES - 1 : 1 ))
make -j"$JOBS" alpha_zero_torch_example
echo "binary: $(find . -name 'alpha_zero_torch_example' -type f)"
