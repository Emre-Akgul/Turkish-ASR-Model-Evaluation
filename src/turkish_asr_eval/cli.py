from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from turkish_asr_eval.datasets import (
    DatasetSpec,
    compute_error_rates,
    extract_audio_input,
    load_hf_dataset,
    parse_dataset_spec,
)
from turkish_asr_eval.engines import UnknownEngineError, available_engines, create_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on Hugging Face datasets."
    )
    parser.add_argument("--engine", required=True, choices=available_engines())
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device")
    parser.add_argument("--compute-type")
    parser.add_argument("--name")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N samples from the split",
    )
    return parser


def engine_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    if args.device is not None:
        options["device"] = args.device
    if args.compute_type is not None:
        options["compute_type"] = args.compute_type
    return options


def make_run_name(engine: str, model: str, spec: DatasetSpec) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw = f"{engine}-{model}-{spec.name}-{spec.split}-{timestamp}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")


def row_count(dataset: Any, spec: DatasetSpec) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        pass

    info = getattr(dataset, "info", None)
    splits = getattr(info, "splits", None)
    if splits is None or spec.split not in splits:
        return None

    return getattr(splits[spec.split], "num_examples", None)


def get_column_names(dataset: Any) -> list[str]:
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        return list(column_names)

    features = getattr(dataset, "features", None)
    if features is not None:
        return list(features.keys())

    return []


def iter_indexed_rows(dataset: Any) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, row in enumerate(dataset):
        yield index, row


def evaluate_row(
    index: int,
    row: dict[str, Any],
    *,
    engine: Any,
    engine_name: str,
    model: str,
    spec: DatasetSpec,
    reference_column: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "engine": engine_name,
        "model": model,
        "dataset": spec.name,
        "hf_dataset": spec.hf_name,
        "dataset_config": spec.config,
        "split": spec.split,
        "audio_column": spec.audio_column,
        "reference_column": spec.reference_column,
        "row_index": index,
    }
    started = time.perf_counter()
    try:
        audio = extract_audio_input(row[spec.audio_column])
        prediction = engine.transcribe(audio)
        record["prediction"] = prediction

        if reference_column is not None:
            reference = str(row[reference_column])
            record["reference"] = reference
            try:
                wer, cer = compute_error_rates(reference, prediction)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["wer"] = wer
                record["cer"] = cer
    except Exception as exc:  # Keep long evaluations from losing prior rows.
        record["prediction"] = ""
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
    return record


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["rows"] += 1
    if "error" in record:
        summary["error_rows"] += 1
    if "wer" in record and "cer" in record:
        summary["scored_rows"] += 1
        summary["_wer_sum"] += record["wer"]
        summary["_cer_sum"] += record["cer"]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    output = {
        "rows": summary["rows"],
        "scored_rows": summary["scored_rows"],
        "error_rows": summary["error_rows"],
    }
    if summary["scored_rows"]:
        output["mean_wer"] = summary["_wer_sum"] / summary["scored_rows"]
        output["mean_cer"] = summary["_cer_sum"] / summary["scored_rows"]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_record(handle: Any, summary: dict[str, Any], record: dict[str, Any]) -> None:
    update_summary(summary, record)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def initial_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": 0,
        "scored_rows": 0,
        "error_rows": 0,
        "_wer_sum": 0.0,
        "_cer_sum": 0.0,
    }
    return summary


def run(args: argparse.Namespace) -> Path:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    spec = parse_dataset_spec(args.dataset)
    dataset = load_hf_dataset(spec, streaming=True)
    column_names = get_column_names(dataset)
    if column_names and spec.audio_column not in column_names:
        raise ValueError(
            f"Audio column '{spec.audio_column}' was not found. Available columns: "
            f"{', '.join(column_names)}"
        )

    if column_names and spec.reference_column not in column_names:
        raise ValueError(
            f"Reference column '{spec.reference_column}' was not found. Available columns: "
            f"{', '.join(column_names)}"
        )
    run_name = args.name or make_run_name(args.engine, args.model, spec)
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.jsonl"

    try:
        engine = create_engine(args.engine, args.model, **engine_options(args))
    except UnknownEngineError:
        raise
    engine.load()

    rows: Iterable[tuple[int, dict[str, Any]]] = iter_indexed_rows(dataset)
    total = row_count(dataset, spec)
    if args.limit is not None:
        rows = islice(rows, args.limit)
        total = min(total, args.limit) if total is not None else args.limit
    summary = initial_summary()
    with output_path.open("w", encoding="utf-8") as handle:
        if args.workers == 1:
            iterator = (
                evaluate_row(
                    index,
                    row,
                    engine=engine,
                    engine_name=args.engine,
                    model=args.model,
                    spec=spec,
                    reference_column=spec.reference_column,
                )
                for index, row in rows
            )
            for record in tqdm(iterator, total=total, unit="sample"):
                write_record(handle, summary, record)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                iterator = executor.map(
                    lambda item: evaluate_row(
                        item[0],
                        item[1],
                        engine=engine,
                        engine_name=args.engine,
                        model=args.model,
                        spec=spec,
                        reference_column=spec.reference_column,
                    ),
                    rows,
                )
                for record in tqdm(iterator, total=total, unit="sample"):
                    write_record(handle, summary, record)

    write_summary(output_dir / f"{run_name}.summary.json", summary)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = run(args)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Wrote results to {output_path}")
    return 0
