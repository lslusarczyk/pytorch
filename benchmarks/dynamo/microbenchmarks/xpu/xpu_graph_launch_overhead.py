import argparse
import time
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


"""
This benchmark demonstrates CPU kernel launch overhead
dominating small-batch XPU workloads and the benefit
of XPUGraph replay.

Expected:
- Speedup increases with depth
- Speedup decreases with width
- Batch size fixed to 1

VTune (Intel VTune Profiler):
    This benchmark uses torch.autograd.profiler.emit_itt() so each autograd op
    emits Intel ITT (Instrumentation and Tracing Technology) ranges while the
    timed eager / xpugraph sections run. profiler.record_function("eager" / "xpugraph")
    adds named regions on top of that.

    How to view in VTune (names vary slightly by VTune version):

    1. GUI: New Project -> Launch Application -> set Python as the app and pass
       this script + args (default model is minimal; e.g. tiny_kernel_storm --iters 100
       --depth 50 --width 128).

    2. Choose an analysis that collects user/ITT instrumentation, e.g. Hotspots
       (User-Mode Sampling) or Threading, and enable ITT / user task collection
       in the analysis properties (look for "ITT", "User API", or
       "Instrumentation and Tracing Technology").

    3. Run the analysis. In the result, open the timeline / Platform view and
       look for ITT task ranges (autograd op names, and regions like "eager",
       "xpugraph", or with --fine-grain-itt: "eager_iter", "xpugraph_iter",
       "xpugraph_synch" per batch).

    4. CLI example (adjust -collect and knobs for your VTune install):
       vtune -collect hotspots -knob collect-user-itt-api=true -- \\
           python xpu_graph_launch_overhead.py --iters 100
           python xpu_graph_launch_overhead.py tiny_kernel_storm --iters 100 --depth 50 --width 128

    Warmup before the ITT-wrapped sections is not annotated; only the measured
    loops inside emit_itt() show dense ITT marks.
"""

class MinimalAddKernel(nn.Module):
    """One trivial XPU op: element-wise add (minimal launch overhead baseline)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class TinyKernelStorm(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def run_eager(model, x, iters, fine_grain_itt):
    for _ in range(50):
        model(x)
    torch.xpu.synchronize()

    start = time.perf_counter()
    with profiler.emit_itt(), profiler.record_function("eager"):    
        if fine_grain_itt:
            for _ in range(iters):
                with profiler.record_function("eager_iter"):
                    model(x)    
        else:
            for _ in range(iters):
                model(x)
    
    with profiler.record_function("eager_synch"):
        torch.xpu.synchronize()
        
    return (time.perf_counter() - start) / iters

def _compare_graph_to_eager(ref_y, static_y, *, rtol=1e-2, atol=1e-2):
    ok = torch.allclose(ref_y, static_y, rtol=rtol, atol=atol)
    if not ok:
        diff = (ref_y - static_y).abs()
        max_abs = diff.max().item()
        denom = ref_y.abs().clamp(min=1e-8)
        max_rel = (diff / denom).max().item()
        raise AssertionError(
            f"graph vs eager mismatch (rtol={rtol}, atol={atol}): "
            f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
        )


def run_xpu_graph(
    model,
    x,
    iters,
    fine_grain_itt,
    ref_y=None,
    batch_size=None,
    *,
    native_recording=False,
):
    g = torch.xpu.XPUGraph(native_recording=native_recording)
    static_x = x.clone()

    for _ in range(10):
        model(static_x)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        static_y = model(static_x)

    if ref_y is not None:
        # On XPU, capture only records; outputs are written on replay(). Replay once
        # so static_y holds the graph result, then compare to eager ref_y.
        #static_x.copy_(x)
        g.replay()
        torch.xpu.synchronize()
        _compare_graph_to_eager(ref_y, static_y)
        print("verify: graph vs eager match (rtol=1e-2, atol=1e-2)")

    if batch_size is None:
        batch_size = iters
    batch_size = max(1, batch_size)

    torch.xpu.synchronize()
    total_time = 0.0
    with profiler.emit_itt(), profiler.record_function("xpugraph"):
        remaining = iters
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            start = time.perf_counter()
            for _ in range(this_batch):
                if fine_grain_itt:
                    with profiler.record_function("xpugraph_iter"):
                        #static_x.copy_(x)
                        g.replay()
                else:
                    #static_x.copy_(x)
                    g.replay()
            if fine_grain_itt:
                with profiler.record_function("xpugraph_synch"):
                    torch.xpu.synchronize()
            else:
                torch.xpu.synchronize()
            end = time.perf_counter()
            total_time += end - start
            remaining -= this_batch

    return total_time / iters

def main():
    parser = argparse.ArgumentParser(
        description=(
            "XPUGraph launch overhead benchmark. "
            "Default model is minimal (single add). "
            "Use tiny_kernel_storm for depth/width stack of Linear+ReLU blocks."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="minimal",
        choices=("minimal", "tiny_kernel_storm"),
        help="minimal (default): one add kernel; tiny_kernel_storm: use --depth/--width.",
    )
    parser.add_argument("--depth", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help=(
            "In the timed XPUGraph section: synchronize every N replay iterations. "
            "Each perf_counter interval covers one batch (N replays + sync); "
            "a final partial batch is timed and synced the same way. "
            "Default: N = --iters (one batch, sync only at end)."
        ),
    )
    parser.add_argument("--fine-grain-itt", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that graph replay produces close results to eager (same input).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on the model (requires Triton with Intel/XPU backend; pip Triton often has none).",
    )
    parser.add_argument(
        "--include-eager",
        action="store_true",
        help="Also run eager timing and print speedup vs graph. Default: graph path only.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help=(
            "Create XPUGraph with SYCL property::graph::enable_native_recording "
            "(Level Zero native graph capture). Requires a capable SYCL/UR stack."
        ),
    )
    args = parser.parse_args()
    batch_size = args.batch if args.batch is not None else args.iters
    if batch_size < 1:
        raise SystemExit("--batch must be >= 1")

    assert torch.xpu.is_available()

    device = "xpu"
    if args.model == "minimal":
        model = MinimalAddKernel().to(device).eval()
        x = torch.randn(1, device=device)
    else:
        model = TinyKernelStorm(args.depth, args.width).to(device).eval()
        x = torch.randn(1, args.width, device=device)
    if args.compile:
        model = torch.compile(model)

    ref_y = None
    if args.verify:
        ref_y = model(x)
        torch.xpu.synchronize()

    if args.include_eager:
        eager_t = run_eager(model, x, args.iters, args.fine_grain_itt)
        graph_t = run_xpu_graph(
            model,
            x,
            args.iters,
            args.fine_grain_itt,
            ref_y=ref_y,
            batch_size=batch_size,
            native_recording=args.native,
        )
        print(f"Eager:     {eager_t * 1000:.3f} ms")
        print(f"XPUGraph:  {graph_t * 1000:.3f} ms")
        print(f"Speedup:   {eager_t / graph_t:.2f}x")
    else:
        graph_t = run_xpu_graph(
            model,
            x,
            args.iters,
            args.fine_grain_itt,
            ref_y=ref_y,
            batch_size=batch_size,
            native_recording=args.native,
        )
        print(f"XPUGraph:  {graph_t * 1000:.3f} ms")

if __name__ == "__main__":
    main()
