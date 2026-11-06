"""Run with python -m metrics score / prepare-gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterator

from .io import checked_path, load_json, read_jsonl, write_json, write_jsonl
from .scorer import ToolCatalog, evaluate
from .records import TimingRecord
from .ranking import aggregate_five_runs, rank_reports
from .replay import load_tool_env, verify_submission


def _different_output(output: Path, *inputs: Path) -> None:
    checked_path(output)
    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("Output path must differ from input files")


def _gold_rows(rows: list[dict], tool_env, data_root: Path) -> Iterator[dict]:
    seen = set()
    # Validate every label before starting any session.
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Missing, invalid or duplicate train id: {sample_id!r}")
        # IDs are used by ToolEnv to locate one sample directory.
        if Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id or ":" in sample_id or sample_id in (".", ".."):
            raise ValueError(f"Unsafe sample id: {sample_id!r}")
        seen.add(sample_id)
        label = row.get("label")
        if not isinstance(label, dict) or not isinstance(label.get("tool_calls"), list):
            raise ValueError(f"{sample_id}: expected train label.tool_calls")
        for call in label["tool_calls"]:
            if (
                not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)
            ):
                raise ValueError(f"{sample_id}: malformed gold tool call")
    if not rows:
        raise ValueError("Train labels must contain at least one sample")
    for row in rows:
        # Exactly one official session for this sample; no direct app-data reads.
        env = tool_env("train", row["id"], root=data_root)
        for call in row["label"]["tool_calls"]:
            result = env.call(call["name"], call["arguments"])
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError(f"{row['id']}: gold call rejected: {call['name']}: {result!r}")
        yield {
            "id": row["id"], "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local LifeMate state + tool-call metrics")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Compare a submission against hashed gold JSONL")
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)
    score.add_argument("--catalog", type=Path, required=True, help="Tool catalog for the same split")
    score.add_argument("--output", type=Path, help="Write summary and per-sample details as JSON")
    score.add_argument("--require-all", action="store_true", help="Reject missing rows instead of scoring them 0")
    score.add_argument("--timings", type=Path, help="Per-gold-ID inference timing/status JSONL; apply 30s limit")
    score.add_argument("--replay", action="store_true", help="Verify calls with official ToolEnv; recomputed hash is authoritative")
    score.add_argument("--data-root", type=Path, help="Required for --replay; contains split directories")
    score.add_argument("--split", choices=("train", "public_test", "private_test"), default="train")
    score.add_argument("--replay-timeout", type=float, default=30.0, help="Verifier deadline, separate from inference timing")
    prepare = commands.add_parser("prepare-gold", help="Replay labeled train calls once via official ToolEnv")
    prepare.add_argument("--data-root", type=Path, required=True, help="Directory containing train/")
    prepare.add_argument("--labels", type=Path, help="Labeled train JSONL subset (default: train/train.jsonl)")
    prepare.add_argument("--output", type=Path, required=True)
    average = commands.add_parser("average-runs", help="Average exactly 5 separately generated score reports")
    average.add_argument("--reports", nargs=5, type=Path, required=True)
    average.add_argument("--output", type=Path, required=True)
    rank = commands.add_parser("rank", help="Rank reports by score, tool F1, then inference time")
    rank.add_argument("--entry", action="append", required=True, metavar="NAME=REPORT.json")
    rank.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            if args.output:
                _different_output(args.output, args.gold, args.submission, args.catalog, *([args.timings] if args.timings else []))
            catalog = ToolCatalog.from_dict(load_json(args.catalog))
            submitted, gold = read_jsonl(args.submission), read_jsonl(args.gold)
            timings = [TimingRecord.from_dict(r) for r in read_jsonl(args.timings)] if args.timings else None
            # Check IDs, shapes and complete timing coverage before any replay.
            report = evaluate(
                submitted, gold, catalog, require_all=args.require_all, timings=timings,
            )
            replay = None
            if args.replay:
                if not args.data_root:
                    raise ValueError("--replay requires --data-root")
                replay = verify_submission(submitted, args.data_root, args.split, args.catalog, replay_timeout_seconds=args.replay_timeout)
                report = evaluate(submitted, gold, catalog, require_all=args.require_all, timings=timings, replays=replay.records)
            elif args.data_root:
                raise ValueError("--data-root in score requires --replay")
            result = report.to_dict()
            if replay:
                result["verification"] = replay.runtime
            if args.output:
                write_json(args.output, result)
            print(json.dumps(report.summary, ensure_ascii=True, indent=2, allow_nan=False))
        elif args.command == "prepare-gold":
            root = checked_path(args.data_root).resolve()
            labels = args.labels or root / "train" / "train.jsonl"
            catalog_path = root / "train" / "train_tools_catalog.json"
            _different_output(args.output, labels, catalog_path)
            rows = read_jsonl(labels)
            write_jsonl(args.output, _gold_rows(rows, load_tool_env(root, "train"), root))
            print(json.dumps({"gold_samples": len(rows), "output": str(args.output)}, ensure_ascii=True))
        elif args.command == "average-runs":
            _different_output(args.output, *args.reports)
            result = aggregate_five_runs([load_json(path) for path in args.reports])
            write_json(args.output, result)
            print(json.dumps(result["summary"], ensure_ascii=True, indent=2, allow_nan=False))
        else:
            entries = []
            sources = []
            for entry in args.entry:
                name, separator, source = entry.partition("=")
                if not separator or not source:
                    raise ValueError("Each --entry must be NAME=REPORT.json")
                sources.append(Path(source))
                entries.append((name, load_json(source)))
            _different_output(args.output, *sources)
            result = rank_reports(entries)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, RecursionError) as error:
        parser.exit(2, f"metrics: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
