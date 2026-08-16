#!/usr/bin/env python3

import argparse
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = int(os.environ.get("RUNS", "3"))
PYTHON = [sys.executable, str(ROOT / "microgpt.py")]
MOJO_SOURCE = ROOT / "microgpt.mojo"
DATASET = ROOT / "input.txt"
DATASET_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"


def execute(command):
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.stderr)
    return time.perf_counter() - start


def peak_rss(command):
    flag = "-l" if sys.platform == "darwin" else "-v"
    result = subprocess.run(
        ["/usr/bin/time", flag, *command],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.stderr)
    if sys.platform == "darwin":
        match = re.search(
            r"^\s*(\d+)\s+maximum resident set size$", result.stderr, re.MULTILINE
        )
        scale = 1
    else:
        match = re.search(
            r"Maximum resident set size \(kbytes\):\s*(\d+)", result.stderr
        )
        scale = 1024
    if not match:
        raise SystemExit("Could not read peak RSS from /usr/bin/time")
    return int(match.group(1)) * scale / 1024**2


def median_runtime(command, runs):
    return statistics.median(execute(command) for _ in range(runs))


def ensure_dataset():
    if not DATASET.exists():
        urllib.request.urlretrieve(DATASET_URL, DATASET)
    _ = DATASET.read_bytes()  # keep dataset I/O out of first-invocation timings


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python and Mojo execution modes."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Measured runs per mode (default: {DEFAULT_RUNS}).",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    mojo = os.environ.get("MOJO") or shutil.which("mojo")
    if not mojo:
        raise SystemExit("mojo was not found on PATH")

    ensure_dataset()
    mojo_run = [mojo, "run", "-O3", str(MOJO_SOURCE)]
    with tempfile.TemporaryDirectory() as directory:
        binary = str(Path(directory) / "microgpt")
        build_time = execute([mojo, "build", "-O3", str(MOJO_SOURCE), "-o", binary])
        python_first = execute(PYTHON)
        mojo_run_first = execute(mojo_run)
        mojo_binary_first = execute([binary])
        python_runtime = median_runtime(PYTHON, args.runs)
        mojo_run_runtime = median_runtime(mojo_run, args.runs)
        mojo_binary_runtime = median_runtime([binary], args.runs)
        python_rss = peak_rss(PYTHON)
        mojo_run_rss = peak_rss(mojo_run)
        mojo_binary_rss = peak_rss([binary])

    mojo_build_and_first = build_time + mojo_binary_first
    break_even = (
        math.ceil(build_time / (python_runtime - mojo_binary_runtime))
        if python_runtime > mojo_binary_runtime
        else None
    )

    print(f"microGPT benchmark ({args.runs} measured runs, median)")
    print(
        f"{'Implementation':30}{'Runtime':>11}{'Speedup':>11}{'Peak RSS':>13}{'RSS/Python':>13}"
    )
    print(
        f"{'Python':30}{python_runtime:>10.3f}s{'1.00x':>11}{python_rss:>10.1f} MB{'1.00x':>13}"
    )
    print(
        f"{'Mojo run (compile + execute)':30}{mojo_run_runtime:>10.3f}s{python_runtime / mojo_run_runtime:>10.2f}x{mojo_run_rss:>10.1f} MB{mojo_run_rss / python_rss:>12.2f}x"
    )
    print(
        f"{'Mojo -O3 binary':30}{mojo_binary_runtime:>10.3f}s{python_runtime / mojo_binary_runtime:>10.2f}x{mojo_binary_rss:>10.1f} MB{mojo_binary_rss / python_rss:>12.2f}x"
    )

    print("\nFirst invocation in this benchmark")
    print(f"{'Python':30}{python_first:>10.3f}s")
    print(f"{'Mojo run (compile + execute)':30}{mojo_run_first:>10.3f}s")
    print(f"{'Mojo build + first execution':30}{mojo_build_and_first:>10.3f}s")
    print(f"\nMojo new-output -O3 build: {build_time:.3f}s")
    print(
        "Optimized-binary break-even:",
        f"{break_even} run(s)" if break_even else "not reached",
    )


if __name__ == "__main__":
    main()
