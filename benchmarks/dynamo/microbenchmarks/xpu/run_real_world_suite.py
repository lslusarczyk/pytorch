"""Run real_world_app.py repeatedly from a CSV suite file; collect JSON latencies and print a table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SUITE = _THIS_DIR / "suite_1.txt"
_REAL_WORLD_APP = _THIS_DIR / "real_world_app.py"
_TCMALLOC = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so"

# (mean_ms, stdev_ms) from repeats, or None if every repeat failed for that column.
LatencyStat = tuple[float, float] | None

# How many times each (suite row × tcmalloc × setup) runs; `--iter` per run is suite_iter // this.
_SUITE_REPEAT_COUNT = 5
_SUITE_REPEAT_COUNT_QUICK = 2

# After sourcing ~/.bash_intel: (bash command line, latency column title). One bash subprocess per entry.
_LS_SETUP_ENV_PYTORCH_ONEAPI: list[tuple[str, str]] = [
    ("ls_setup_env_pytorch_oneapi", "baseline"),
    ("ls_setup_env_pytorch_oneapi 2", "PR"),
]


def _parse_suite_row(row: list[str]) -> tuple[int, int, str]:
    """Return (tcmalloc_mode, iter, extra_cli_string). tcmalloc_mode: 0/1 = once without preload; 2 = without then with."""
    parts = [c.strip() for c in row if c.strip() != ""]
    if len(parts) < 2:
        raise ValueError(f"Suite row needs at least 2 non-empty fields, got {row!r}")
    mode = int(parts[0])
    if mode not in (0, 1, 2):
        raise ValueError(f"First field must be 0, 1, or 2, got {mode}")
    if len(parts) >= 3:
        it = int(parts[1])
        extra = " ".join(parts[2:])
    else:
        rest = parts[1]
        head_tail = rest.split(None, 1)
        it = int(head_tail[0])
        extra = head_tail[1].strip() if len(head_tail) > 1 else ""
    return mode, it, extra


def _tcmalloc_runs(mode: int) -> list[bool]:
    """False = no LD_PRELOAD (unset); True = LD_PRELOAD=tcmalloc."""
    if mode == 2:
        return [False, True]
    return [False]


def _extra_args_bash_array(extra: str) -> str:
    toks = shlex.split(extra, posix=True) if extra.strip() else []
    if not toks:
        return "EXTRA=()"
    inner = " ".join(shlex.quote(t) for t in toks)
    return f"EXTRA=({inner})"


def _build_bash_body(
    *,
    use_tcmalloc: bool,
    iter_n: int,
    extra: str,
    out_json_no_compile: str,
    out_json_compile: str,
    ls_setup_env_line: str,
) -> str:
    ld_block = (
        f'export LD_PRELOAD={shlex.quote(_TCMALLOC)}'
        if use_tcmalloc
        else "unset LD_PRELOAD || true"
    )
    extra_line = _extra_args_bash_array(extra)
    py_script = shlex.quote(str(_REAL_WORLD_APP.resolve()))
    out0 = shlex.quote(out_json_no_compile)
    out1 = shlex.quote(out_json_compile)
    iter_s = str(int(iter_n))
    # Do not use `set -u`: Intel/oneAPI setup scripts often reference $1 etc. when unset.
    return f"""set -eo pipefail
if [ -f "${{HOME}}/.bash_intel" ]; then
    . "${{HOME}}/.bash_intel" || {{ echo "Error: Failed to source ${{HOME}}/.bash_intel. Aborting." >&2; exit 1; }}
else
    echo "Error: Required environment file ${{HOME}}/.bash_intel not found. Aborting." >&2
    exit 1
fi

{ls_setup_env_line}

pt_activate
cd "${{HOME}}/src" || {{ echo "Error: cd to ${{HOME}}/src failed." >&2; exit 1; }}
{ld_block}
{extra_line}
taskset -c 0,2,4,6,8 python3 {py_script} --device xpu --iter {iter_s} "${{EXTRA[@]}}" --output-json {out0}
taskset -c 0,2,4,6,8 python3 {py_script} --device xpu --iter {iter_s} "${{EXTRA[@]}}" --compile --output-json {out1}
"""


def _run_one_bash(body: str) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, encoding="utf-8"
    ) as f:
        f.write("#!/usr/bin/env bash\n" + body)
        path = f.name
    try:
        os.chmod(path, 0o700)
        p = subprocess.run(["bash", path], check=False)
        return int(p.returncode)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _read_latency_json(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return float(data["latency_ms"])


def _mean_std_ms(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    m = statistics.mean(samples)
    if len(samples) < 2:
        return m, 0.0
    return m, statistics.stdev(samples)


def _format_latency_mean_std(mean_ms: float, std_ms: float) -> str:
    """Mean in ms (4 significant figures) ± sample stdev (2 significant figures)."""
    return f"{format(mean_ms, '.4g')} ± {format(std_ms, '.2g')}"


def _format_latency_cell(st: LatencyStat) -> str:
    if st is None:
        return ""
    return _format_latency_mean_std(st[0], st[1])


def _format_speedup_cell(lat: LatencyStat, base: LatencyStat) -> str:
    if lat is None or base is None:
        return ""
    lat_m, lat_s = lat
    base_m, base_s = base
    return _format_speedup_vs_baseline(lat_m, lat_s, base_m, base_s)


def _format_speedup_vs_baseline(
    lat_m: float,
    lat_s: float,
    base_m: float,
    base_s: float,
) -> str:
    """S = (base/lat)*100 - 100 (%). σ_S = 100*σ_R with R=base/lat, independent σ on means."""
    if lat_m == 0.0:
        return "n/a"
    t_b = base_s / lat_m
    t_l = base_m * lat_s / (lat_m * lat_m)
    sigma_r = math.sqrt(t_b * t_b + t_l * t_l)
    sigma_pct = 100.0 * sigma_r
    pct = (base_m / lat_m) * 100.0 - 100.0
    return f"{format(pct, '.4g')}% ± {format(sigma_pct, '.2g')}%"


def _build_results_table_cells(
    rows: list[tuple[str, tuple[LatencyStat, ...]]],
    latency_col_titles: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Column headers and body rows as preformatted strings (plain + GFM)."""
    n = len(latency_col_titles)
    headers: list[str] = ["run"]
    for i in range(n):
        headers.append(latency_col_titles[i])
        if i > 0:
            headers.append("speedup")
    body: list[list[str]] = []
    for desc, lats in rows:
        base = lats[0]
        cells = [desc]
        for i in range(n):
            cells.append(_format_latency_cell(lats[i]))
            if i > 0:
                cells.append(_format_speedup_cell(lats[i], base))
        body.append(cells)
    return headers, body


def _print_plain_aligned_table(headers: list[str], body: list[list[str]]) -> None:
    sep = "  "
    ncol = len(headers)
    widths = [len(headers[j]) for j in range(ncol)]
    for row in body:
        for j in range(ncol):
            widths[j] = max(widths[j], len(row[j]))
    hdr_parts = [f"{headers[0]:<{widths[0]}}"]
    rule_parts = ["-" * widths[0]]
    for j in range(1, ncol):
        hdr_parts.append(f"{headers[j]:>{widths[j]}}")
        rule_parts.append("-" * widths[j])
    print(sep.join(hdr_parts))
    print(sep.join(rule_parts))
    for row in body:
        parts = [f"{row[0]:<{widths[0]}}"]
        for j in range(1, ncol):
            parts.append(f"{row[j]:>{widths[j]}}")
        print(sep.join(parts))


def _md_table_cell(text: str) -> str:
    """GFM table cell: avoid breaking on | and newlines (GitHub-flavored Markdown)."""
    return text.replace("\n", " ").replace("|", "&#124;")


def _print_github_markdown_table(headers: list[str], body: list[list[str]]) -> None:
    """Pipe table per https://docs.github.com/en/get-started/writing-on-github/.../basic-writing-and-formatting-syntax"""
    print()
    print(
        "### GitHub Markdown (paste into a comment)\n"
        "<!-- GFM tables: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax -->"
    )
    h_line = "| " + " | ".join(_md_table_cell(h) for h in headers) + " |"
    div = "| " + " | ".join("---" for _ in headers) + " |"
    print(h_line)
    print(div)
    for row in body:
        print("| " + " | ".join(_md_table_cell(c) for c in row) + " |")


def _print_results_table(
    rows: list[tuple[str, tuple[LatencyStat, ...]]],
    latency_col_titles: list[str],
) -> None:
    if not latency_col_titles:
        return
    headers, body = _build_results_table_cells(rows, latency_col_titles)
    _print_plain_aligned_table(headers, body)
    _print_github_markdown_table(headers, body)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run real_world_app.py for each suite row (see suite CSV format). "
            "For each entry in _LS_SETUP_ENV_PYTORCH_ONEAPI (bash line, column title), "
            "runs one shell with that setup line, then two benchmarks (no --compile, then --compile); "
            f"each configuration is repeated ({_SUITE_REPEAT_COUNT}×, or {_SUITE_REPEAT_COUNT_QUICK}× "
            "with --quick); per-run --iter is suite iter divided by that repeat count. "
            "The table shows mean latency (4 sig. fig.) ± sample stdev (2 sig. fig.) per column, "
            "plus speedup (mean vs first column) and propagated σ (quotient rule on R=baseline/lat) "
            "after each non-baseline latency column. Results are printed as plain text, then again "
            "as a GitHub-flavored Markdown pipe table for pasting into comments. "
            "Failed benchmark runs leave cells blank and the suite continues; exit status is non-zero "
            "if any run failed."
        )
    )
    ap.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"CSV suite file (default: {_DEFAULT_SUITE})",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use only the first two non-empty suite rows, --iter 50 for each, "
            f"and {_SUITE_REPEAT_COUNT_QUICK} repeats (instead of {_SUITE_REPEAT_COUNT}); "
            "per-run --iter is suite iter ÷ repeat count."
        ),
    )
    args = ap.parse_args()

    suite_path = args.suite.resolve()
    if not suite_path.is_file():
        print(f"Error: suite file not found: {suite_path}", file=sys.stderr)
        return 1
    if not _REAL_WORLD_APP.is_file():
        print(f"Error: real_world_app.py not found: {_REAL_WORLD_APP}", file=sys.stderr)
        return 1
    if not _LS_SETUP_ENV_PYTORCH_ONEAPI:
        print(
            "Error: _LS_SETUP_ENV_PYTORCH_ONEAPI must list at least one (command, title) pair.",
            file=sys.stderr,
        )
        return 1
    for i, ent in enumerate(_LS_SETUP_ENV_PYTORCH_ONEAPI):
        if (
            not isinstance(ent, tuple)
            or len(ent) != 2
            or not ent[0].strip()
            or not ent[1].strip()
        ):
            print(
                f"Error: _LS_SETUP_ENV_PYTORCH_ONEAPI[{i}] must be a non-empty (str, str) tuple.",
                file=sys.stderr,
            )
            return 1
    if _SUITE_REPEAT_COUNT < 1 or _SUITE_REPEAT_COUNT_QUICK < 1:
        print(
            "Error: _SUITE_REPEAT_COUNT and _SUITE_REPEAT_COUNT_QUICK must be >= 1.",
            file=sys.stderr,
        )
        return 1

    rows_out: list[tuple[str, tuple[LatencyStat, ...]]] = []
    quick_max_rows = 2 if args.quick else None
    n_repeats = (
        _SUITE_REPEAT_COUNT_QUICK if args.quick else _SUITE_REPEAT_COUNT
    )
    data_rows_done = 0
    any_run_failed = False

    with open(suite_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not c.strip() for c in row):
                continue
            if quick_max_rows is not None and data_rows_done >= quick_max_rows:
                break
            try:
                mode, iter_from_file, extra = _parse_suite_row(row)
            except (ValueError, IndexError) as e:
                print(f"Error: suite line {line_no}: {e}", file=sys.stderr)
                return 1
            iter_n = 50 if args.quick else iter_from_file
            iter_per_run = max(1, iter_n // n_repeats)
            desc_base = f"iter={iter_per_run} {extra}".strip()
            if args.quick:
                desc_base = f"[quick] {desc_base}".strip()

            for use_tc in _tcmalloc_runs(mode):
                tc_label = "tcmalloc" if use_tc else "no tcmalloc"
                lat_nc_each: list[LatencyStat] = []
                lat_c_each: list[LatencyStat] = []
                for si, (setup_cmd, _title) in enumerate(_LS_SETUP_ENV_PYTORCH_ONEAPI):
                    samples_nc: list[float] = []
                    samples_c: list[float] = []
                    for rep in range(n_repeats):
                        with tempfile.NamedTemporaryFile(
                            suffix=".json", delete=False, prefix="real_world_suite_nc_"
                        ) as jf0:
                            json_path_nc = jf0.name
                        with tempfile.NamedTemporaryFile(
                            suffix=".json", delete=False, prefix="real_world_suite_c_"
                        ) as jf1:
                            json_path_c = jf1.name
                        try:
                            body = _build_bash_body(
                                use_tcmalloc=use_tc,
                                iter_n=iter_per_run,
                                extra=extra,
                                out_json_no_compile=json_path_nc,
                                out_json_compile=json_path_c,
                                ls_setup_env_line=setup_cmd,
                            )
                            rc = _run_one_bash(body)
                            if rc != 0:
                                any_run_failed = True
                                print(
                                    f"Warning: bash exited {rc} (suite line {line_no}, "
                                    f"{tc_label}, setup[{si}]={setup_cmd!r}, "
                                    f"repeat {rep + 1}/{n_repeats}); skipping this repeat.",
                                    file=sys.stderr,
                                )
                            else:
                                try:
                                    samples_nc.append(
                                        _read_latency_json(Path(json_path_nc))
                                    )
                                    samples_c.append(
                                        _read_latency_json(Path(json_path_c))
                                    )
                                except (OSError, json.JSONDecodeError, KeyError) as e:
                                    any_run_failed = True
                                    print(
                                        f"Warning: JSON read failed (suite line {line_no}, "
                                        f"{tc_label}, setup[{si}]={setup_cmd!r}, "
                                        f"repeat {rep + 1}/{n_repeats}): {e}",
                                        file=sys.stderr,
                                    )
                        finally:
                            for p in (json_path_nc, json_path_c):
                                try:
                                    os.unlink(p)
                                except OSError:
                                    pass
                    if samples_nc:
                        lat_nc_each.append(_mean_std_ms(samples_nc))
                    else:
                        lat_nc_each.append(None)
                    if samples_c:
                        lat_c_each.append(_mean_std_ms(samples_c))
                    else:
                        lat_c_each.append(None)

                rows_out.append(
                    (
                        f"{desc_base} | {tc_label} | no --compile",
                        tuple(lat_nc_each),
                    )
                )
                rows_out.append(
                    (
                        f"{desc_base} | {tc_label} | --compile",
                        tuple(lat_c_each),
                    )
                )

            data_rows_done += 1

    _print_results_table(
        rows_out, [t[1] for t in _LS_SETUP_ENV_PYTORCH_ONEAPI]
    )

    return 1 if any_run_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
