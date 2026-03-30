"""XPU graph benchmark for a stacked transformer decode (vLLM-style autoregressive loop).

Each iteration dispatches one forward pass, samples a token via argmax (device→host copy), and reports
median and p99 latency in milliseconds. The device→host transfer simulates the per-token dependency in
real autoregressive decoding.
"""

import argparse
import statistics
import time
from typing import Callable

import torch

from model import TransformerStack

# Benchmark configuration
D_MODEL  = 4096
N_HEADS  = 32
KV_LEN   = 1024
BATCH    = 3
N_LAYERS    = 16
ITERATIONS  = 200

DTYPE           = torch.float16
DEVICE          = torch.device("xpu")
WARMUP_ITERS    = 5


def _warmup(fn: "Callable[[], object]", warmup_iters: int) -> None:
    """Call fn warmup_iters times, then synchronize."""
    for _ in range(warmup_iters):
        fn()
    torch.xpu.synchronize()


def _summarize(batch_timings_s: list[float]) -> tuple[float, float]:
    """Return (median_ms, p99_ms) of wall-clock times in milliseconds."""
    timings_ms = [t * 1e3 for t in batch_timings_s]
    p99 = statistics.quantiles(timings_ms, n=100, method="inclusive")[98] if len(timings_ms) >= 2 else timings_ms[0]
    return statistics.median(timings_ms), p99


def run_eager(
    model: torch.nn.Module,
    x: torch.Tensor,
    iterations: int,
    warmup_iters: int = WARMUP_ITERS,
) -> tuple[float, float]:
    """Return (median_ms, p99_ms) across all iterations."""
    print("Warming up eager execution...")
    _warmup(lambda: model(x, position=0), warmup_iters)
    print("Timing eager execution...")
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        logits = model(x, position=0)
        logits.argmax(-1).cpu()  # device→host copy
        timings.append(time.perf_counter() - t0)
    return _summarize(timings)

def run_xpu_graph(
    model: torch.nn.Module,
    x: torch.Tensor,
    iterations: int,
    warmup_iters: int = WARMUP_ITERS,
) -> tuple[float, float]:
    """Return (median_ms, p99_ms) across all iterations."""
    print("Capturing XPU graph...")
    g = torch.xpu.XPUGraph()
    static_x = x.clone()
    print("Warming up before capture...")
    _warmup(lambda: model(static_x, position=0), warmup_iters)

    print("Capturing...")
    with torch.xpu.graph(g):
        static_y = model(static_x, position=0)
    torch.xpu.synchronize()

    # Input is fixed across replays in a graph benchmark — copy once before timing.
    print("Copying input before timing...")
    static_x.copy_(x)
    torch.xpu.synchronize()

    print("Warming up graph replay...")
    _warmup(g.replay, warmup_iters)

    print("Timing graph replay...")
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        g.replay()                    # submits entire graph in ~microseconds
        static_y.argmax(-1).cpu()     # device→host copy
        timings.append(time.perf_counter() - t0)
    return _summarize(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="XPU graph benchmark")
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Number of benchmark iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=N_LAYERS,
        help="Number of stacked decode layers (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=WARMUP_ITERS,
        help="Number of warmup iterations (default: %(default)s)",
    )
    args = parser.parse_args()
    iterations = args.iterations
    n_layers = args.n_layers
    warmup_iters = args.warmup_iters

    model = TransformerStack(
        n_layers=n_layers,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        kv_len=KV_LEN,
        batch=BATCH,
        dtype=DTYPE,
        device=DEVICE,
    ).to(device=DEVICE, dtype=DTYPE)
    model.eval()

    x = torch.randn(BATCH, 1, D_MODEL, dtype=DTYPE, device=DEVICE)

    eager_median, eager_p99 = run_eager(model, x, iterations=iterations, warmup_iters=warmup_iters)
    graph_median, graph_p99 = run_xpu_graph(model, x, iterations=iterations, warmup_iters=warmup_iters)

    print(f"{'d_model':>12s}: {D_MODEL}")
    print(f"{'n_heads':>12s}: {N_HEADS}")
    print(f"{'kv_len':>12s}: {KV_LEN}")
    print(f"{'batch':>12s}: {BATCH}")
    print(f"{'n_layers':>12s}: {n_layers}")
    print(f"{'iterations':>12s}: {iterations}")
    print(f"{'warmup_iters':>12s}: {warmup_iters}")
    print(f"{'dtype':>12s}: {DTYPE}")
    print(f"{'device':>12s}: {DEVICE}")
    print()
    print(f"{'':25s}  {'median':>12s}  {'p99':>12s}")
    print(f"{'Eager':25s}  {eager_median:>11.3f}ms  {eager_p99:>11.3f}ms")
    print(f"{'XPU graph':25s}  {graph_median:>11.3f}ms  {graph_p99:>11.3f}ms")
    print(f"Speedup (median):  {eager_median / graph_median:.3f}x")
    print(f"Speedup (p99):     {eager_p99 / graph_p99:.3f}x")


if __name__ == "__main__":
    main()
