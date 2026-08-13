# Third-party notices

## OpenSpiel-derived patch files (Apache-2.0 only)

`scripts/instrument_vpevaluator.patch` and `scripts/az_device_game_example.patch`
contain portions of [OpenSpiel](https://github.com/google-deepmind/open_spiel)
source (Copyright DeepMind Technologies Limited), licensed under the
Apache License, Version 2.0. These two files are therefore available under
**Apache-2.0 only** — the repository's MIT option does not apply to them.
The full Apache-2.0 text is in [LICENSE-APACHE](LICENSE-APACHE).

`scripts/setup_openspiel_cpp.sh` additionally restores two upstream directories
(`open_spiel/libtorch` and `open_spiel/libnop`, five build files) content-identical
from OpenSpiel's own history at build time; the restored content is likewise
OpenSpiel's, under Apache-2.0.
