# microGPT in Python and Mojo

> A small, dependency-free character-level GPT, implemented twice to compare a clear Python reference with native Mojo.

## Why this repository exists

I am a data scientist and AI engineer, so most of my work happens at a high level: analyzing data, building models, and deploying them. Still, I have always been interested in what happens beneath those abstractions, especially when it comes to performance.

Faster machine-learning workloads can reduce cost, energy use, and latency, while enabling larger models. But optimization is also appealing in itself: understanding where resources go, removing overhead, and pushing hardware closer to its limits.

The challenge is that performance often comes at the cost of convenience. Python is productive and expressive, but much of its speed comes from lower-level code in C or C++, creating a two-language stack.

Mojo is built around the idea that this trade-off is not inevitable. It aims to combine Python-like usability with systems-level control and performance. With Mojo 1.0 now available, this repository is my hands-on exploration of that idea through translating and optimizing a familiar machine-learning program.

## Implementations

This repository contains an educational character-level GPT implemented in two programming languages to compare performance and usability:

| File | Description |
| --- | --- |
| [`microgpt.py`](microgpt.py) | Andrej Karpathy's scalar-autograd microGPT reference |
| [`microgpt.mojo`](microgpt.mojo) | Native Mojo port using an explicit reverse-mode tape |
| [`benchmark.py`](benchmark.py) | Reproducible runtime and peak-memory comparison |
| [`FUNFACTS.md`](FUNFACTS.md) | Notes on Mojo, MLIR, compile-time specialization, and ownership |

This is intentionally a CPU-focused, pedagogical implementation rather than a production training framework. Both versions use one transformer layer, a 16-dimensional embedding, four attention heads, a context length of 16, 1,000 training steps, and fixed random seeds.

The languages use different random-number generators, so their exact losses and generated names are not expected to match. The implementations also use different autograd representations: Python builds an object-based computation graph, while Mojo uses an append-only indexed tape.

## Requirements

- macOS on Apple silicon, or Linux on x86-64/AArch64
- A C linker (`xcode-select --install` on macOS, or GCC on Linux)
- [Pixi](https://pixi.sh/)

Pixi installs Python 3.11 and Mojo 1.0.0 from the committed lockfile; no separate environment activation is required.

## Quick start

```bash
git clone https://github.com/Ktemmerman/microgpt-python-mojo.git
cd microgpt-python-mojo

pixi run python
pixi run mojo
```

On the first run, either implementation downloads Karpathy's names dataset to `input.txt` if it is not already present. Each version trains for 1,000 steps and generates 20 names.

Build and run the optimized Mojo executable:

```bash
pixi run mojo-optimized
```

## Benchmark

Compare Python, Mojo run mode, and a prebuilt optimized Mojo executable:

```bash
pixi run benchmark
```

The default uses three measured runs per mode. To choose another count:

```bash
pixi run benchmark --runs 5
```

The report includes:

- median end-to-end runtime and speedup;
- peak resident memory;
- first-invocation time within the benchmark;
- new-output Mojo `-O3` build time; and
- the number of runs needed to amortize compilation.

Wall time includes process startup, 1,000 training steps, and sampling. Mojo run mode includes compilation, while the optimized-binary result excludes compilation. The first-invocation figures are not cold-machine measurements because operating-system and compiler caches may already be warm.

This benchmark compares these two educational implementations; it does not isolate language performance. Any speedup reflects the languages, runtimes, compilers, and different autograd representations together. Close background applications before recording results.

## Development tasks

```bash
pixi run check       # Compile-check Python and Mojo
pixi run mojo-build  # Build build/microgpt_mojo with -O3
pixi run clean       # Remove generated files
```

## Attribution

The original Python algorithm is by [Andrej Karpathy](https://github.com/karpathy) and comes from his [microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The source attribution is preserved in [`microgpt.py`](microgpt.py).
