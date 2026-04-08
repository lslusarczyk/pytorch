# shows real influence on production inference without using Eager Mode
# model -> torch.compile -> with or without CUDA graphs
# these are image classificatoin models
#
# VTune (Intel VTune Profiler):
#   With --emit-itt, torch.autograd.profiler.emit_itt() marks each autograd op in the timed
#   section; record_function(...) always adds named regions (eager, graph_replay, ...).
#   Use --fine-grain-itt for per-iteration regions. Enable ITT in the VTune analysis properties.

import argparse
import contextlib
import logging
import sys
import time

import timm
import torch
import torch.autograd.profiler as profiler
import torchvision.models as models


def _configure_stdout_logger(name: str) -> logging.Logger:
    """Timestamped INFO logs to stdout; flush after each record."""
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False

    class _FlushStreamHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    handler = _FlushStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    return log


parser = argparse.ArgumentParser(description="Testing CUDA/SYCL Graphs.")
parser.add_argument(
    "--graphs",
    "--graph",
    action="store_true",
    help=("Use device capture/replay graphs (CUDA or XPU)."),
)
parser.add_argument("--profiler", action="store_true", help="Use profiler.")
parser.add_argument("--logs", action="store_true", help="Log if cuda graphs are used or not.")  # alternatively set TORCH_LOGS="inductor,cuda_graphs"
parser.add_argument("--autographs", "--autograph", action="store_true", help="Automatic use of graphs in torch.compile and with kernel fusions")
parser.add_argument("--compile", action="store_true", help="Call compile on model")
parser.add_argument("--iter", type=int, default=3000, help="Number of iterations")
parser.add_argument(
    "--batch",
    type=int,
    default=None,
    metavar="N",
    help=(
        "In the timed section: synchronize every N iterations (graph replay or eager forward). "
        "Each batch runs N steps then syncs; a final partial batch syncs the same way. "
        "Default: N = --iter (one batch, sync only at end)."
    ),
)
parser.add_argument("--device", type=str, choices=['xpu', 'cuda'], default='xpu', help="Which backend to use.")
parser.add_argument("--model", type=str, choices=['resnet', 'transformer', 'retina', 'vit'], default='resnet', help="Which model to use.")
parser.add_argument("--retina-size", type=int, default=224, help="retina input size")
parser.add_argument(
    "--vit-size",
    type=int,
    default=16,
    help="ViT spatial size H=W (must be divisible by patch size 16 for vit_*_patch16_*).",
)
parser.add_argument("--skip-consistency-check", action="store_true", help="Skip graph vs eager consistency check when using --graphs.")
parser.add_argument(
    "--fine-grain-itt",
    action="store_true",
    help="Per-iteration record_function regions and labeled sync for VTune (see module docstring).",
)
parser.add_argument(
    "--emit-itt",
    action="store_true",
    help="Wrap timed loops with profiler.emit_itt() (dense per-op ITT for VTune). Default: off.",
)
args = parser.parse_args()

batch_size = args.batch if args.batch is not None else args.iter
if batch_size < 1:
    raise SystemExit("--batch must be >= 1")

logger = _configure_stdout_logger("real_world_app")

if args.graphs and args.device == "cuda" and args.model == "retina":
    logger.warning(
        "RetinaNet eval postprocess_detections (NMS, masking, etc.) uses ops CUDA graph capture does not allow"
    )

def _itt_ctx():
    return profiler.emit_itt() if args.emit_itt else contextlib.nullcontext()


def _compare_outputs(ref, graph_out, *, rtol=1e-1, atol=1e-1, name="output"):
    """Compare reference (eager) and graph outputs. Raises AssertionError on mismatch."""
    if isinstance(ref, torch.Tensor):
        ok = (
            torch.allclose(ref, graph_out, rtol=rtol, atol=atol)
            if ref.is_floating_point()
            else torch.equal(ref, graph_out)
        )
        if not ok:
            diff = (ref - graph_out).abs()
            max_abs_diff = diff.max().item()
            denom = ref.abs().clamp(min=1e-8)
            max_rel_diff = (diff / denom).max().item()
            raise AssertionError(
                f"{name}: graph output does not match eager (rtol={rtol}, atol={atol}). "
                f"max_abs_diff={max_abs_diff:.3e}, max_rel_diff={max_rel_diff:.3e}"
            )
        return
    if isinstance(ref, (list, tuple)):
        assert len(ref) == len(graph_out), f"{name}: length mismatch {len(ref)} vs {len(graph_out)}"
        for i, (r, g) in enumerate(zip(ref, graph_out)):
            if isinstance(r, dict):
                assert set(r.keys()) == set(g.keys()), f"{name}[{i}]: dict keys mismatch"
                for k in r:
                    if isinstance(r[k], torch.Tensor):
                        if r[k].is_floating_point():
                            ok = torch.allclose(r[k], g[k], rtol=rtol, atol=atol)
                            if not ok:
                                d = (r[k] - g[k]).abs()
                                max_abs = d.max().item()
                                denom = r[k].abs().clamp(min=1e-8)
                                max_rel = (d / denom).max().item()
                                raise AssertionError(
                                    f"{name}[{i}].{k}: graph does not match eager, "
                                    f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
                                )
                        else:
                            assert torch.equal(r[k], g[k]), f"{name}[{i}].{k}: graph does not match eager"
                    else:
                        assert r[k] == g[k], f"{name}[{i}].{k}: graph does not match eager"
            else:
                _compare_outputs(r, g, rtol=rtol, atol=atol, name=f"{name}[{i}]")
        return
    raise TypeError(f"{name}: unsupported output type {type(ref)}")


def _materialize_detection_transform_stats_on_device(model: torch.nn.Module, device: str) -> None:
    """GeneralizedRCNNTransform keeps image_mean / image_std as Python lists. During forward,
    torchvision builds mean/std via torch.as_tensor(..., device=image.device), which can
    trigger a CPU→device copy that CUDA graph capture rejects unless the source is pinned.
    Storing mean/std as tensors already on the target device avoids that copy inside capture."""
    root = getattr(model, "_orig_mod", model)
    t = getattr(root, "transform", None)
    if t is None or not hasattr(t, "image_mean") or not hasattr(t, "image_std"):
        return
    dev = torch.device(device)
    dtype = next(root.parameters()).dtype

    def _tensor_on_dev(x):
        if isinstance(x, torch.Tensor):
            if x.device == dev and x.dtype == dtype:
                return x
            return x.to(device=dev, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=dev)

    t.image_mean = _tensor_on_dev(t.image_mean)
    t.image_std = _tensor_on_dev(t.image_std)


torch.set_float32_matmul_precision('high')

class Backend:
    def __init__(self):
        if args.device == 'cuda':
            self.synchronize = torch.cuda.synchronize
            self.create_graph = torch.cuda.CUDAGraph
            self.graph = torch.cuda.graph
            self.empty_cache = torch.cuda.empty_cache
        elif args.device == 'xpu':
            assert torch.xpu.is_available()
            self.synchronize = torch.xpu.synchronize
            self.create_graph = torch.xpu.XPUGraph
            self.graph = torch.xpu.graph
            self.empty_cache = torch.xpu.empty_cache
        else:
            raise RuntimeError(f"unknown backend {args.device}")


backend = Backend()

if args.logs:
    import torch._logging

    # TORCHINDUCTOR_CUDA_GRAPHS=1 <- force graphs ??
    # https://docs.pytorch.org/docs/stable/generated/torch._logging.set_logs.html
    torch._logging.set_logs(graph_breaks=True)

if args.model == 'resnet':
    # CNN - convolutional net
    logger.info("resnet50 model to device")
    model = models.resnet50().to(args.device).eval()
elif args.model == 'transformer':
    # vit: Vision Transformer
    # base: model size
    # patch16; spliting image into 16x16 patches (14x14 patches grid)
    # 224: expected image resolution
    # like in NLP: layernorm, attention, matmul, softmax, matmul, mlp. gelu - many small kernels
    
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(args.device).eval()
elif args.model == 'retina':
    model = models.detection.retinanet_resnet50_fpn(pretrained=False, weights=None).to(args.device).eval()
elif args.model == 'vit':
    # timm ViT expects (B, C, H, W). Default checkpoint is 224; set img_size to match --vit-size.
    vs = args.vit_size
    if vs % 16 != 0:
        raise SystemExit("--vit-size must be divisible by 16 for vit_*_patch16_*")
    logger.info("create model")
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=vs,
    )
    logger.info("move model to device")
    model = model.to(args.device)
    logger.info("set model to eval")
    model = model.eval()
else:
    raise RuntimeError(f"unknown model {args.model}")

logger.info("model created")

if args.autographs:
    logger.info("compile it with reduce-overhead")
    model = torch.compile(model, mode="reduce-overhead")
elif args.compile:
    logger.info("compile it")
    model = torch.compile(model)
else:
    logger.info("skipping compile")

# look for https://docs.pytorch.org/docs/stable/generated/torch.compile.html
# torch._inductor.list_mode_options()
# see options, in particular
# triton.cudagraphs which will reduce the overhead of python with CUDA graphs

logger.info("fill random")  # one image in a batch 3 channels, 224x224 pixels
if args.model == 'retina':
    size =  args.retina_size
    static_x = torch.randn(3,size,size,device=args.device)
    x = [static_x]
    is_list_input = True
elif args.model == 'vit':
    size = args.vit_size
    x = torch.randn(1, 3, size, size, device=args.device)
    is_list_input = False
else:
    x = torch.randn(1, 3, 224, 224, device=args.device)
    is_list_input = False

schedule = torch.profiler.schedule(
    wait=10, warmup=20, active=60, repeat=1)

with torch.inference_mode():
    logger.info("warmup")
    for _ in range(50):
        model(x)

    logger.info("synchronize after warmup")
    backend.synchronize()
    backend.empty_cache()

    N = args.iter

    if args.graphs:
        if args.model == "retina":
            _materialize_detection_transform_stats_on_device(model, args.device)
        logger.info("prepare graph")
        g = backend.create_graph()

        if is_list_input:
            static_x = [x[0].clone()]
        else:
            static_x = x.clone()

        with backend.graph(g):
            static_y = model(static_x)

        if not args.skip_consistency_check:
            ref_y = model(x)
            g.replay()
            backend.synchronize()
            _compare_outputs(ref_y, static_y, name="graph vs eager")
            logger.info("consistency check passed (graph vs eager)")

        logger.info("graph case: start %s iterations measured", N)
        start = time.time()

        def _one_graph_replay_iter():
            if is_list_input:
                static_x[0].copy_(x[0])
            else:
                static_x.copy_(x)
            g.replay()

        synch_name = "graph_replay_synch"

        if args.profiler:
            with torch.profiler.profile(schedule=schedule, acc_events=True, activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                with _itt_ctx(), profiler.record_function("graph_replay"):
                    remaining = N
                    while remaining > 0:
                        this_batch = min(batch_size, remaining)
                        for _ in range(this_batch):
                            if args.fine_grain_itt:
                                with profiler.record_function("graph_replay_iter"):
                                    _one_graph_replay_iter()
                            else:
                                _one_graph_replay_iter()
                            prof.step()
                        if args.fine_grain_itt:
                            with profiler.record_function(synch_name):
                                backend.synchronize()
                        else:
                            backend.synchronize()
                        remaining -= this_batch
        else:
            with _itt_ctx(), profiler.record_function("graph_replay"):
                remaining = N
                while remaining > 0:
                    this_batch = min(batch_size, remaining)
                    for _ in range(this_batch):
                        if args.fine_grain_itt:
                            with profiler.record_function("graph_replay_iter"):
                                _one_graph_replay_iter()
                        else:
                            _one_graph_replay_iter()
                    if args.fine_grain_itt:
                        with profiler.record_function(synch_name):
                            backend.synchronize()
                    else:
                        backend.synchronize()
                    remaining -= this_batch

    else:

        logger.info("non-graph case, start %s iterations measured", N)
        start = time.time()

        synch_name = "eager_synch"
        if args.profiler:
            with torch.profiler.profile(schedule=schedule, acc_events=True, activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                with _itt_ctx(), profiler.record_function("eager"):
                    remaining = N
                    while remaining > 0:
                        this_batch = min(batch_size, remaining)
                        for _ in range(this_batch):
                            if args.fine_grain_itt:
                                with profiler.record_function("eager_iter"):
                                    model(x)
                            else:
                                model(x)
                            prof.step()
                        if args.fine_grain_itt:
                            with profiler.record_function(synch_name):
                                backend.synchronize()
                        else:
                            backend.synchronize()
                        remaining -= this_batch
        else:
            with _itt_ctx(), profiler.record_function("eager"):
                remaining = N
                while remaining > 0:
                    this_batch = min(batch_size, remaining)
                    for _ in range(this_batch):
                        if args.fine_grain_itt:
                            with profiler.record_function("eager_iter"):
                                model(x)
                        else:
                            model(x)
                    if args.fine_grain_itt:
                        with profiler.record_function(synch_name):
                            backend.synchronize()
                    else:
                        backend.synchronize()
                    remaining -= this_batch

end_time = time.time()

if args.profiler:
    logger.info("\n%s", prof.key_averages().table(sort_by="self_cuda_time_total"))

logger.info("Latency: %.3f msec", 1000 * (end_time - start) / N)

