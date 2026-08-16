# Sizing the compute

## Result

For the V1 network, CUDA is already 53× faster than CPU at the smallest measured batch
and reaches a plateau near 21,000 rows/s. The curve is device- and network-specific; it
is a diagnostic, not a portable recommendation.

| Batch | CUDA rows/s | CPU rows/s | CUDA / CPU |
|---:|---:|---:|---:|
| 32 | 13,857 | 261 | 53× |
| 64 | 15,278 | 263 | 58× |
| 128 | 14,295 | 246 | 58× |
| 256 | 19,266 | 232 | 83× |
| 512 | 20,236 | 201 | 101× |
| 1,024 | 20,655 | 203 | 102× |
| 2,048 | 20,879 | 203 | 103× |

CUDA improves non-monotonically and then plateaus; CPU declines after batch 64. CUDA is
faster throughout the measured range, so the crossover itself is not measured.

## What this measurement is for

This is a pure forward-pass measurement of the shared w256 d8 benchmark network. It
separates device behavior from engine scheduling and bounds the inference throughput
available to either stack. reinfors' compiled-callback curve is reported with
[engine configuration](configuring-the-engines.md), because compilation is specific to
its operating path.
