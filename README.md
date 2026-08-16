# microGPT in Python and Mojo

> A small, dependency-free character-level GPT, implemented twice to compare a clear Python reference with native Mojo.

Built and co-authored with [Pi](https://github.com/badlogic/pi-mono).

## Why this repository exists

I am a data scientist: most of my work is analyzing data and training machine-learning models, not writing low-level code tuned for specific hardware. Still, knowing that those models could run much more efficiently has always bothered me. Performance matters because it can reduce infrastructure costs and energy use, lower latency, and create room for larger or more capable models. There is also a deep beauty in efficient software, and writing fast code is fun.

Python is excellent because of its readability, ecosystem, and role as a universal connector, while C++ is compelling for its control and performance. In practice, however, writing and deploying both languages is difficult. The common compromise is a Python interface backed by C or C++, which introduces a two-language development stack. Mojo is tackling exactly this tension: it aims to combine Python-like usability with the control and performance of a systems language.

Mojo 1.0 has just been released, making this a good moment to explore how that promise holds up in practice. This repository is a personal, hands-on experiment to understand what Mojo can do, how the language and toolchain work, and how it feels to translate a familiar machine-learning program from Python. It is an exploration rather than a definitive performance study.

## Project status

The educational implementation is complete and runnable end to end:

- **Python reference:** scalar reverse-mode autograd, training, and sampling.
- **Mojo port:** the same model structure and hyperparameters, backed by a compact index-based autograd tape.
- **Optimized build:** the Mojo version can be compiled as a native `-O3` executable.
- **Benchmark tooling:** compares runtime, speedup, peak memory, cold-start time, build time, and break-even point.
- **Project checks:** both Python and Mojo currently compile successfully with the locked environment.

This is intentionally a CPU-focused, pedagogical implementation rather than a production training framework. Both versions use one transformer layer, a 16-dimensional embedding, four attention heads, a context length of 16, 1,000 training steps, and fixed random seeds.

## Implementations

| File | Description |
| --- | --- |
| [`microgpt.py`](microgpt.py) | Andrej Karpathy's scalar-autograd microGPT reference |
| [`microgpt.mojo`](microgpt.mojo) | Native Mojo port using an explicit reverse-mode tape |
| [`benchmark.py`](benchmark.py) | Reproducible runtime and peak-memory comparison |
| [`FUNFACTS.md`](FUNFACTS.md) | Notes on Mojo, MLIR, compile-time specialization, and ownership |

The implementations share the same architecture and training configuration. Their random-number generators differ, so exact losses and generated names are not expected to match line for line.

## Requirements

- macOS on Apple silicon, or Linux on x86-64/AArch64
- A C linker (`xcode-select --install` on macOS, or GCC on Linux)
- [Pixi](https://pixi.sh/)

Pixi installs Python 3.11 and Mojo 1.0.0 from the committed lockfile; no separate environment activation is required.

## Quick start

```bash
git clone <repository-url>
cd <repository-directory>

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
- cold-to-result time;
- clean Mojo `-O3` build time; and
- the number of runs needed to amortize compilation.

Wall time includes process startup, 1,000 training steps, and sampling. Mojo run mode includes compilation, while the optimized-binary result excludes compilation. Close background applications before recording benchmark results.

## Development tasks

```bash
pixi run check       # Compile-check Python and Mojo
pixi run mojo-build  # Build build/microgpt_mojo with -O3
pixi run clean       # Remove generated files
```

## Attribution

The original Python algorithm is by [Andrej Karpathy](https://github.com/karpathy) and comes from his [microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The source attribution is preserved in [`microgpt.py`](microgpt.py).
