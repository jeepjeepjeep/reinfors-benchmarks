# reinfors-benchmarks

Reproducible systems benchmarks for
[reinfors](https://github.com/jeepjeepjeep/reinfors), with OpenSpiel as the V1
comparison. This repository contains the experiment code, frozen campaign specs, raw
artifacts and published interpretation.

## V1 results at a glance

All measurements use chess, the same AlphaZero-style network and search budget, and one
AWS g5.2xlarge host (NVIDIA A10G, four physical CPU cores).

| Question | Result |
|---|---|
| Which configurations should be compared? | **OpenSpiel:** 256 actors, batch 256. **reinfors:** 512 games, two groups. |
| How did the selected configurations perform in the sizing sweep? | **236.2 states/s** for OpenSpiel and **265.7 states/s** for reinfors. This selects the configurations; it is not the final training comparison. |
| How do they compare during matched two-hour training? | **264.6 states/s** for OpenSpiel and **289.9 states/s** for reinfors, with meaning reinfors collected 9.8% more training data per two-hour round. |
| Which trained agent is stronger? | The reinfors-trained nets score **0.605 ± 0.020** over 300 paired games (**+74 Elo**, 95% CI +46 to +103). |

Every number is backed by the manifests and telemetry under `published/v1/`.

## Read the results

1. **[The comparison](docs/the-comparison.md)** — matched training throughput and the
   trained-agent head-to-head.
2. **[Configuring the engines](docs/configuring-the-engines.md)** — how the operating
   points were selected and what reinfors' throughput levers contribute.
3. **[Sizing the compute](docs/sizing-the-compute.md)** — the device-level batch curve.

The [methodology](docs/methodology.md) defines the measurement rules. The
[design comparison](docs/design-differences.md) explains the architectural differences
behind the results. Neither is required to read the headline tables.

## Scope

V1 asks a narrow question: how much throughput does reinfors preserve against a mature
C++ implementation while retaining its modular Rust/Python boundary? It is not a
general ranking of either library. Results apply to this workload, network, host and
software environment; core-count and multi-device scaling are not measured.

## Repository map

| Path | Contents |
|---|---|
| `experiments/` | Measurement harnesses, runner, shared protocol and tests. |
| `experiments/specs/` | Frozen V1 experiment matrices. |
| `runs/` | Untracked, append-only campaign evidence. |
| `published/` | Tracked artifacts supporting published numbers. |
| `scripts/` | Environment setup, plots and OpenSpiel patches. |
| `docs/` | Results, interpretation and reference material. |

See **[Reproducing the benchmarks](docs/reproducing.md)** for environments, host setup,
campaign order and commands. The full V1 campaign is roughly **48 hours of sequential
measurement** on the dedicated GPU host — the sizing sweeps, training legs and
head-to-head are not something to re-run quickly on a local machine. OpenSpiel build
deviations are recorded separately in [the upstream notes](docs/openspiel_upstream_notes.md).

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. OpenSpiel-derived patch files are Apache-2.0 only; see
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES.md).
