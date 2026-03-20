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
       this script + args (e.g. --iters 100 --depth 50 --width 128).

    2. Choose an analysis that collects user/ITT instrumentation, e.g. Hotspots
       (User-Mode Sampling) or Threading, and enable ITT / user task collection
       in the analysis properties (look for "ITT", "User API", or
       "Instrumentation and Tracing Technology").

    3. Run the analysis. In the result, open the timeline / Platform view and
       look for ITT task ranges (autograd op names, and regions like "eager",
       "xpugraph", or "eager_iter" / "xpugraph_iter" with --fine-grain-itt).

    4. CLI example (adjust -collect and knobs for your VTune install):
       vtune -collect hotspots -knob collect-user-itt-api=true -- \\
           python xpu_graph_launch_overhead.py --iters 100 --depth 50 --width 128

    Warmup before the ITT-wrapped sections is not annotated; only the measured
    loops inside emit_itt() show dense ITT marks.
"""

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

    with profiler.emit_itt(), profiler.record_function("eager"):
        start = time.perf_counter()
        for _ in range(iters):
            if fine_grain_itt:
                with profiler.record_function("eager_iter"):
                    model(x)
            else:
                model(x)
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


def run_xpu_graph(model, x, iters, fine_grain_itt, ref_y=None):
    g = torch.xpu.XPUGraph()
    static_x = x.clone()

    for _ in range(10):
        model(static_x)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        static_y = model(static_x)

    if ref_y is not None:
        # On XPU, capture only records; outputs are written on replay(). Replay once
        # so static_y holds the graph result, then compare to eager ref_y.
        static_x.copy_(x)
        g.replay()
        torch.xpu.synchronize()
        _compare_graph_to_eager(ref_y, static_y)
        print("verify: graph vs eager match (rtol=1e-2, atol=1e-2)")

    torch.xpu.synchronize()
    with profiler.emit_itt(), profiler.record_function("xpugraph"):
        start = time.perf_counter()
        for _ in range(iters):
            if fine_grain_itt:
                with profiler.record_function("xpugraph_iter"):
                    static_x.copy_(x)
                    g.replay()
            else:
                static_x.copy_(x)
                g.replay()
        torch.xpu.synchronize()
        end = time.perf_counter()

    return (end - start) / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1000)
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
    args = parser.parse_args()

    assert torch.xpu.is_available()

    device = "xpu"
    model = TinyKernelStorm(args.depth, args.width).to(device).eval()
    if args.compile:
        model = torch.compile(model)
    x = torch.randn(1, args.width, device=device)

    ref_y = None
    if args.verify:
        ref_y = model(x)
        torch.xpu.synchronize()

    if args.include_eager:
        eager_t = run_eager(model, x, args.iters, args.fine_grain_itt)
        graph_t = run_xpu_graph(model, x, args.iters, args.fine_grain_itt, ref_y=ref_y)
        print(f"Eager:     {eager_t * 1000:.3f} ms")
        print(f"XPUGraph:  {graph_t * 1000:.3f} ms")
        print(f"Speedup:   {eager_t / graph_t:.2f}x")
    else:
        graph_t = run_xpu_graph(model, x, args.iters, args.fine_grain_itt, ref_y=ref_y)
        print(f"XPUGraph:  {graph_t * 1000:.3f} ms")

if __name__ == "__main__":
    main()
