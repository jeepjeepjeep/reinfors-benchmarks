# Design differences

Where the two stacks' numbers differ, the causes are identifiable structural choices —
each sensible for its system's goals. This page describes both architectures and traces
each measured difference to its design origin, in both directions: several of
OpenSpiel's choices are the right ones for a reference research library, and reinfors'
choices cost it elsewhere (generality, per-game latency).

## The two architectures

**OpenSpiel (C++ libtorch AlphaZero)** runs *independent actor threads*, each playing
its own game with its own MCTS. Actors submit inference requests to a central service
whose batcher gathers up to `inference_batch_size` requests per forward. The MCTS
evaluator asks two questions per expanded node — a value query and a policy-prior
query, as separate calls — and an LRU cache in front of the network merges the pair
into one forward and serves cross-search repeats. The learner runs periodically: when
enough new states arrive it takes the device and performs a full sweep of buffer-sized
minibatches, then actors resume against refreshed weights.

**reinfors** runs a *lockstep pooled search*: all `n_games` games advance together, and
each simulation round gathers their uncached leaves into one callback that returns
policy and value from one forward. The inference cache is position-keyed and
independent of call structure. The learner is caller-owned Python running concurrently
(records stream out per collected batch); weight refreshes are explicit and clear the
cache at round boundaries. Optionally, games split into two groups so one group's tree
work overlaps the other's inference —
[grouped collection](configuring-the-engines.md).

## Consequence 1: batch formation

OpenSpiel's asynchronous batcher needs enough independently progressing actors to fill
a large inference batch — batch size and CPU parallelism are **coupled**. reinfors
stages up to one fresh leaf per active game in a synchronized round — **decoupled**.
Neither batch size is simply the configured count: cache hits, terminal simulations and
deduplication remove rows, while larger actor/game counts increase per-game latency and
leave more work in flight at the deadline. The
[sizing grids](configuring-the-engines.md) measure whether fuller batches outweigh
those completion costs — and this coupling difference is why the answer differs by
stack, and why the whole comparison is
[bounded to the few-core regime](the-comparison.md#scope-of-the-claim).

## Consequence 2: one inference question per node, or two

reinfors' callback contract returns both heads in one forward. OpenSpiel's evaluator
interface separates value and prior queries — a clean interface for a library that also
serves rollout-based evaluators without networks — and relies on its cache to merge the
pair. With the cache on (its intended condition) the merge works and the difference
mostly vanishes; the structural residue is cache-shaped rather than compute-shaped
(entries, lookups and eviction pressure per node differ). This is also why **caches are
architecture, not a matched knob**: comparing with caches *disabled* is not a "clean"
condition — cache-off roughly doubles OpenSpiel's forwards per node while leaving
reinfors untouched, so a "row" stops meaning the same unit of work on the two sides.
Each stack runs its own cache design at equal capacity instead
([the comparison](the-comparison.md)).

## Consequence 3: continuous versus burst training

The reinfors learner trains continuously beside collection; self-play runs against
near-current weights, and GPU time interleaves at fine grain. OpenSpiel trains in
periodic sweeps, so actors play against weights up to one sweep stale. Burst training
simplifies synchronization but introduces device contention and staleness; continuous
training keeps weights fresher while sharing the process with a Python training loop.
The two learners amortize differently, so the comparison verifies matched
gradient-samples per state from telemetry rather than assuming it
([the comparison](the-comparison.md)).

## Consequence 4: what a deadline does to in-flight work

Both stacks lose in-flight games at the hard kill; how much is in flight is a design
consequence — proportional to concurrent games and per-game latency. See the
[methodology](methodology.md) for why completed-game states/s is the selection metric.

## Trade-offs

| dimension | OpenSpiel | reinfors |
|---|---|---|
| heterogeneous play | Per-seat `Bot` composition inside its run infrastructure, including evaluation actors and rollout baselines. Batching still requires evaluator integration. | Per-player networks and frozen opponents stay inside `Engine`; evaluation matches mixing searched and external agents run through `Arena` with pooled batched search; fully caller-driven loops use `Env`, outside engine batching and telemetry. |
| new search integration | A new bot can use the actor loop directly; batched network service requires its evaluator queue. | The standard policy seam uses normal collection, including under grouped collection (group workers run any policy's search opaquely). |
| per-game latency | Small actor counts can favor individual-game latency. | Throughput-oriented lockstep batches can increase individual-game latency. |
| training ownership | The C++ learner is self-contained. | The caller-owned Python learner is flexible but shares the process during concurrent collection. |

The scope this page's explanations live under is
[the comparison's](the-comparison.md#scope-of-the-claim): the published results bound
the claim; they do not rank the libraries generally.
