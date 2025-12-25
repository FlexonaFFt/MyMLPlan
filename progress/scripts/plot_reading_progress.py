#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


PointSeries = List[Tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reading progress (%) for multiple books."
    )
    parser.add_argument(
        "--input",
        default="progress/data/reading.json",
        help="Path to reading progress JSON file.",
    )
    parser.add_argument(
        "--output",
        default="progress/output/plots/reading_progress.png",
        help="Path to save the PNG plot.",
    )
    parser.add_argument(
        "--title",
        default="Reading Progress (%)",
        help="Plot title.",
    )
    parser.add_argument(
        "--book-ids",
        nargs="+",
        help="Limit plotting to the specified book ids.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def load_data(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_range(entry: Dict, total_pages: int, book_id: str) -> bool:
    required_keys = {"from", "to", "minutes"}
    if not required_keys.issubset(entry):
        warn(f"Book '{book_id}': missing keys in entry {entry}.")
        return False

    page_from = entry["from"]
    page_to = entry["to"]
    if not isinstance(page_from, int) or not isinstance(page_to, int):
        warn(f"Book '{book_id}': page numbers must be integers in entry {entry}.")
        return False
    if page_from < 1 or page_to < page_from or page_to > total_pages:
        warn(
            f"Book '{book_id}': page range {page_from}-{page_to} is outside 1..{total_pages}."
        )
        return False
    return True


def detect_overlaps(entries: Iterable[Dict], book_id: str) -> None:
    ranges = sorted(((entry["from"], entry["to"]) for entry in entries), key=lambda x: (x[0], x[1]))
    previous = None
    for start, end in ranges:
        if previous and start <= previous[1]:
            warn(
                f"Book '{book_id}': overlapping page ranges {previous[0]}-{previous[1]} and "
                f"{start}-{end}."
            )
        previous = (start, end)


def aggregate_progress(book: Dict) -> PointSeries:
    book_id = book.get("id", "<unknown>")
    title = book.get("title", book_id)
    total_pages = book.get("total_pages", 0)
    if total_pages <= 0:
        warn(f"Book '{book_id}': total_pages must be > 0. Skipping.")
        return []

    valid_entries = []
    for entry in book.get("log", []):
        if not validate_range(entry, total_pages, book_id):
            continue
        valid_entries.append(entry)

    if not valid_entries:
        warn(f"Book '{book_id}' has no valid log entries.")
        return []

    detect_overlaps(valid_entries, book_id)

    cumulative_pages = 0
    series: PointSeries = []

    for entry in valid_entries:
        pages = entry["to"] - entry["from"] + 1
        remaining = max(0, total_pages - cumulative_pages)
        pages_used = min(pages, remaining)

        if pages_used < pages:
            warn(
                f"Book '{book_id}': truncating entry {entry['from']}-{entry['to']} "
                f"to fit remaining pages ({remaining})."
            )

        entry_percent = (pages_used / total_pages) * 100
        cumulative_pages += pages_used
        cumulative_percent = (cumulative_pages / total_pages) * 100
        cumulative_percent = min(cumulative_percent, 100.0)
        series.append((cumulative_percent, entry_percent))

    if series and series[-1][0] >= 100.0:
        warn(f"Book '{book_id}' reached or exceeded 100% cumulative progress (capped at 100).")

    return series


def plot_progress(
    book_series: List[Tuple[str, str, PointSeries]], title: str, output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    max_y = 0.0
    for book_id, book_title, series in book_series:
        if not series:
            continue
        x_vals, y_vals = zip(*series)
        plt.plot(x_vals, y_vals, marker="o", label=book_title)
        max_y = max(max_y, max(y_vals))

    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))
    if max_y <= 0:
        y_max = 10
    else:
        y_max = min(100, max(10, ceil(max_y * 1.1 / 5) * 5))
    y_step = 5 if y_max <= 50 else 10
    plt.ylim(0, y_max)
    plt.yticks(range(0, int(y_max) + 1, y_step))
    plt.xlabel("Completion (%)")
    plt.ylabel("Entry Contribution (%)")
    plt.title(title or "Reading Progress (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        warn(f"Input file not found: {input_path}")
        return 1

    data = load_data(input_path)
    books = data.get("books", [])

    if args.book_ids:
        requested = set(args.book_ids)
        available = {book.get("id") for book in books}
        missing = requested - available
        if missing:
            warn(f"Book ids not found: {', '.join(sorted(missing))}")
        books = [book for book in books if book.get("id") in requested]

    if not books:
        warn("No books to plot after filtering.")
        return 1

    series_per_book: List[Tuple[str, str, PointSeries]] = []
    for book in books:
        series = aggregate_progress(book)
        series_per_book.append((book.get("id", "<unknown>"), book.get("title", ""), series))

    plotted_any = any(series for _, _, series in series_per_book)
    if not plotted_any:
        warn("No valid progress data to plot.")
        return 1

    plot_progress(series_per_book, args.title, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
