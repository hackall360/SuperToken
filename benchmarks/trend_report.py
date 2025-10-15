"""Utilities for turning benchmark JSON outputs into plots and Markdown tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrendPoint:
    timestamp: datetime
    source: Path
    tokens: int
    bpe_wall: float
    unigram_wall: float

    @property
    def label(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:%M")

    @property
    def bpe_tokens_per_second(self) -> float | None:
        if self.bpe_wall <= 0:
            return None
        return self.tokens / self.bpe_wall

    @property
    def unigram_tokens_per_second(self) -> float | None:
        if self.unigram_wall <= 0:
            return None
        return self.tokens / self.unigram_wall


def _load_point(path: Path) -> TrendPoint:
    """Parse a benchmark JSON snapshot into a :class:`TrendPoint`.

    Parameters
    ----------
    path:
        Path to a ``benchmark_*.json`` file produced by the benchmarking
        harness.

    Returns
    -------
    TrendPoint
        Structured representation of the raw metrics contained in ``path``.

    Notes
    -----
    The payload is expected to contain ``timestamp``, ``corpus.tokens``,
    ``bpe.wall_time_s``, and ``unigram.wall_time_s`` fields. When the timestamp
    is missing or uses an unknown format we fall back to the file's mtime.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp_raw = payload.get("timestamp")
    try:
        timestamp = datetime.strptime(timestamp_raw, "%Y%m%dT%H%M%SZ")
    except Exception:  # pragma: no cover - defensive fallback for unexpected formats
        timestamp = datetime.utcfromtimestamp(path.stat().st_mtime)
    tokens = int(payload["corpus"]["tokens"])
    bpe_wall = float(payload["bpe"].get("wall_time_s", 0.0))
    unigram_wall = float(payload["unigram"].get("wall_time_s", 0.0))
    return TrendPoint(
        timestamp=timestamp,
        source=path,
        tokens=tokens,
        bpe_wall=bpe_wall,
        unigram_wall=unigram_wall,
    )


def load_history(inputs: Iterable[Path]) -> list[TrendPoint]:
    """Load and sort a collection of benchmark points.

    Parameters
    ----------
    inputs:
        Iterable of :class:`pathlib.Path` objects pointing at benchmark JSON
        snapshots.

    Returns
    -------
    list[TrendPoint]
        Benchmark points sorted chronologically and secondarily by filename to
        provide a stable ordering.
    """
    points = [_load_point(path) for path in inputs]
    points.sort(key=lambda p: (p.timestamp, p.source.name))
    return points


def _format_value(value: float | None) -> str:
    """Render floating point metrics for inclusion in Markdown tables.

    Parameters
    ----------
    value:
        Numeric value to render. ``None`` is treated as unavailable data.

    Returns
    -------
    str
        Human-friendly string, formatted to two decimal places or ``"n/a"``
        when ``value`` is ``None``.
    """
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def render_table(points: Sequence[TrendPoint], output: Path) -> Path:
    """Write a Markdown table summarising benchmark runs.

    Parameters
    ----------
    points:
        Ordered sequence of :class:`TrendPoint` items to tabulate.
    output:
        Destination :class:`pathlib.Path` whose contents will be replaced with
        a Markdown ``.md`` document.

    Returns
    -------
    Path
        The ``output`` path, allowing for method chaining in calling code.

    Notes
    -----
    The table contains token counts, wall-clock durations, and derived tokens
    per second metrics for both the BPE and unigram trainers. When ``points``
    is empty the file contains the sentence ``_No benchmarks found._`` for
    easier downstream rendering.
    """
    headers = [
        "Timestamp",
        "Tokens",
        "BPE wall (s)",
        "BPE tokens/s",
        "Unigram wall (s)",
        "Unigram tokens/s",
    ]
    rows: list[list[str]] = []
    for point in points:
        rows.append(
            [
                point.label,
                f"{point.tokens:,}",
                _format_value(point.bpe_wall),
                _format_value(point.bpe_tokens_per_second),
                _format_value(point.unigram_wall),
                _format_value(point.unigram_tokens_per_second),
            ]
        )
    line = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    body = "\n".join(" | ".join(row) for row in rows) if rows else "_No benchmarks found._"
    output.write_text("\n".join([line, separator, body]), encoding="utf-8")
    return output


def render_plot(points: Sequence[TrendPoint], output: Path) -> Path:
    """Create a PNG line chart visualising throughput over time.

    Parameters
    ----------
    points:
        Ordered sequence of benchmark points that provide tokens-per-second
        values for the BPE and unigram trainers.
    output:
        Destination :class:`pathlib.Path` where a PNG image will be written.

    Returns
    -------
    Path
        The ``output`` path, allowing callers to reference the saved image.

    Notes
    -----
    The module configures matplotlib to use the ``Agg`` backend so the plot can
    be rendered in headless environments. When ``points`` is empty an empty
    file is produced so downstream automation can detect the absence of data.
    """
    if not points:
        output.write_bytes(b"")
        return output
    labels = [point.label for point in points]
    indices = range(len(points))
    bpe = [point.bpe_tokens_per_second or 0.0 for point in points]
    unigram = [point.unigram_tokens_per_second or 0.0 for point in points]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(indices, bpe, marker="o", label="GPUBPETrainer tokens/s")
    ax.plot(indices, unigram, marker="o", label="GPUUnigramTrainer tokens/s")
    ax.set_xticks(list(indices))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Tokens per second")
    ax.set_xlabel("Benchmark run")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the trend reporting utility.

    Returns
    -------
    argparse.ArgumentParser
        Parser defining ``--input`` (directory of ``benchmark_*.json`` files),
        ``--output-dir`` (destination directory), ``--table-name``, and
        ``--plot-name`` options. All paths are interpreted relative to the
        current working directory when invoked from the shell.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="artifacts/benchmarks",
        help="Directory containing benchmark_*.json snapshots",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/trends",
        help="Directory where the plot and table should be written",
    )
    parser.add_argument(
        "--table-name",
        default="trend_table.md",
        help="File name for the rendered Markdown table",
    )
    parser.add_argument(
        "--plot-name",
        default="trend_plot.png",
        help="File name for the rendered PNG chart",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for generating Markdown and PNG trend reports.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When ``None`` (the
        default) ``argparse`` reads arguments from :data:`sys.argv`.

    Notes
    -----
    The CLI can be invoked directly, for example::

        python -m benchmarks.trend_report \
            --input artifacts/benchmarks \
            --output-dir artifacts/benchmarks/trends

    The command expects ``benchmark_*.json`` files in ``--input`` and writes a
    Markdown table, a PNG chart, and a ``trend_manifest.json`` manifest into
    ``--output-dir``. The manifest records resolved input paths and the derived
    metrics, which makes integration with automation pipelines easier.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist")

    json_paths = sorted(input_dir.glob("benchmark_*.json"))
    points = load_history(json_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / args.table_name
    plot_path = output_dir / args.plot_name

    render_table(points, table_path)
    render_plot(points, plot_path)

    manifest = {
        "inputs": [str(path) for path in json_paths],
        "table": str(table_path),
        "plot": str(plot_path),
        "points": [
            {
                "timestamp": point.timestamp.isoformat(),
                "tokens": point.tokens,
                "bpe_wall_s": point.bpe_wall,
                "bpe_tokens_per_s": point.bpe_tokens_per_second,
                "unigram_wall_s": point.unigram_wall,
                "unigram_tokens_per_s": point.unigram_tokens_per_second,
            }
            for point in points
        ],
    }
    manifest_path = output_dir / "trend_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
