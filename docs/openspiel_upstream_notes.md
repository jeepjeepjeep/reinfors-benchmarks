# OpenSpiel upstream dossier — material for the eventual issue/PR

Everything we learned getting OpenSpiel's C++ libtorch AlphaZero building and performing on a
GPU box (AWS g5.2xlarge, Ubuntu 22.04, A10G), 2026-07-31. All claims below are measured or
reproduced, with commits/SHAs named. Our working recipe: `scripts/setup_openspiel_cpp.sh`.

## Pin policy

The comparison targets OpenSpiel **as maintained today**: a pinned master snapshot
built from source with CUDA libtorch, including upstream's own performance fixes. The
snapshot is taken from the upstream tip and, before any publication run, re-verified
against current master — if commits have landed touching the measured subsystems
(`algorithms/alpha_zero_torch`, the game, the evaluator/bot surfaces), the benchmark
re-pins and rebuilds rather than publishing against a superseded target. The pin in
force (commit + patch hashes + applied-diff hash) is recorded in `open_spiel_cpp/PIN`,
verified by preflight, and carried in every run manifest. The two required
interventions are content-preserving and documented below: restoring build glue master
deletes while still referencing, and measurement instrumentation plus a device flag in
example code.

## Issue inventory (current master, snapshot 112b7770 / 2026-07-17)

1. **libtorch/libnop build glue deleted; path unbuildable since 2025-10-07.**
   Commit `86fe553c` ("ttt-fixes", an internal-sync commit about tic-tac-toe examples)
   deleted `open_spiel/libtorch/CMakeLists.txt`, `open_spiel/libnop/CMakeLists.txt`, their
   two integration tests, and a `.gitignore` — while `open_spiel/CMakeLists.txt` still
   `add_subdirectory`'s both under `OPEN_SPIEL_BUILD_WITH_LIBTORCH/LIBNOP=ON`. Result:
   configure fails ("does not contain a CMakeLists.txt"). Reproduced on today's master with
   a fully-provisioned tree (`find_package(Torch)` succeeds first — it is only the 13 lines
   of wrapper glue that are missing). Unreported upstream as of 2026-07 (no matching issue).
   **Fix = restore the five files from `86fe553c^`** (what our setup script does).

2. **No CI coverage of the libtorch path — the root cause of everything here.**
   `.github/workflows/actions.yml`: no job sets `OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON`.
   This is how (1) went unnoticed for 9+ months, and how the 5-year GPU staging bug (below)
   survived. **Highest-value suggestion: one build-only CI job with CPU libtorch (~200 MB)
   + their own torch_integration_test.** Without it, any restoration is one internal sync
   away from being deleted again.

3. **Ubuntu 22.04 clang (14 AND 15) frontend SEGV compiling `games/dou_dizhu/dou_dizhu.cc`.**
   Deterministic, solo-reproducible, any stack size; crash while parsing (site ~line 169,
   which is innocuous — trigger is elsewhere in the TU/headers, arrived between d15d49f8
   and master; the only dou_dizhu diff is a trivial one-liner, so suspect header/abseil
   interaction). g++ 11.4 compiles the file cleanly; one g++ object links fine into the
   clang build (same Itanium ABI + libstdc++). CI never sees this: matrix is
   ubuntu-24.04/26.04 + macos-14 (default clang-18+).

4. **install.md has rotted vs reality.** It recommends "Ubuntu 22.04, Debian 10, or later"
   and claims clang ">= 7.0.0" / g++ >= 9.2 suffice, tests "Python 3.7-3.10". Measured:
   their recommended OS's entire clang toolchain (<=15) cannot compile master (item 3);
   CI actually runs 24.04+/Python 3.12-3.14. We did NOT deviate from their advertised
   config — we used it and found it broken. **Docs fix: align supported-platform and
   compiler claims with CI, or fix (3).**

5. **`cached_clone` serves stale dependency versions** (`open_spiel/scripts/install.sh:103`).
   The download cache is keyed by directory basename only; the requested `-b <version>` is
   ignored whenever a cache entry exists (`cp -r` returns whatever was cached). Bit us
   concretely: upstream bumped `OPEN_SPIEL_ABSL_VERSION` 20250127.1 -> 20250814.1 (master
   REQUIRES the new abseil MutexLock API), and the cache silently restored the old one ->
   baffling "no matching constructor for absl::MutexLock" errors. **Fix: key cache as
   `<name>-<ref>`, or verify the cached checkout matches the requested ref.** Good separate
   small PR.

## Historical (already fixed upstream — do NOT re-PR, but cite as motivation)

- **5-year GPU staging bug in `vpnet.cc`** (from the 2020 contribution, PR #319, until
  2025-10-10): input tensors allocated ON DEVICE and filled element-wise (~1300 tiny CUDA
  ops/row for chess) + per-element `.item()` output reads. We measured **~9.3 ms/row**,
  width/batch-independent: their GPU was strictly slower than their own CPU at any
  realistic net size (chess w128 d8: 134 rows/s GPU vs 269 CPU, as shipped). Fixed by
  `9972442` (2025-10-10, "Improve performance of Tensor object creation" — from_blob
  staging; landed 3 days AFTER the build glue was deleted, i.e. author built internally)
  and PR #1488 (2026-03, NoGradGuard + batched output extraction; issue #1487).
  Our independent diagnosis + backport for the pinned era: `fix_vpnet_gpu_staging.patch`
  (retired from the tree now that master carries upstream's own fix; in git history)
  (kept for the as-shipped-era measurements; NOT applied to master builds).
- **Measured effect of their fixes** (chess w128 d8, 8 actors, A10G): 134.6 -> ~4,700
  rows/s (**~35x**), GPU now 17x their CPU. Strong PR-motivation numbers: master's code is
  good; it just cannot be built externally (items 1-3).

## PR strategy (agreed 2026-07-31)

- **PR 1 (code)**: restore the five glue files. Zero behavior. Ask maintainers explicitly
  whether the deletion was intentional deprecation; if so the alternative is removing the
  root references + marking the path deprecated in docs (currently it is neither maintained
  nor deprecated — stranded).
- **Same PR or issue (docs)**: toolchain note (known-good: 24.04+ default clang, or g++;
  22.04 clang-14/15 crash on dou_dizhu.cc) + refresh install.md claims (item 4).
- **Issue (structural)**: propose the CPU-libtorch CI job (item 2).
- **PR 2 (small, independent)**: cached_clone version-keying (item 5).
- **Testing evidence to cite**: no CI validates this path, so reviewers will ask. We have:
  scripted clean-box build recipe (Ubuntu 22.04 = their recommended OS), binary runs chess
  self-play on CUDA at measured throughput, before/after numbers spanning their own fixes,
  every failure minimally reproduced (solo compiles, configure-only runs, API-dated
  commits).

## Our environment decisions (for the methods section)

- **Staying on Ubuntu 22.04** (decided 2026-07-31): it is their RECOMMENDED OS per
  install.md, the DL AMI's driver/toolkit convenience is real, and the toolchain cannot
  move benchmark numbers (hot path = prebuilt libtorch 2.3.0+cu121 kernels, identical
  bytes; our compiled share of wall is 1-2%). Workarounds encoded in the setup script:
  clang-15 (14 crashes), dou_dizhu.cc pre-built with g++ (15 crashes too), CUDA 12.8
  steering for libtorch's libnvToolsExt (removed from CUDA >= 12.9 toolkits).
- Local deltas vs pristine master in benchmark builds, exhaustively: (a) the five restored
  glue files [build-only]; (b) toolchain choices above [build-only]; (c)
  `scripts/instrument_vpevaluator.patch` — measurement counters + `[inst]` stderr lines
  [source change, measurement-only, disclosed]. Upstream's inference/training code paths
  are 100% as-maintained.
