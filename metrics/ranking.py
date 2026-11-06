"""Five-run averaging and competition tie-break ordering."""

from collections import defaultdict
from math import fsum, isfinite


def _summary(report: dict) -> dict:
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict) or summary.get("metric_version") != 2 or summary.get("score_scale") != 100:
        raise ValueError("Expected a metrics v2 report on the official 0-100 scale")
    if type(summary.get("num_samples")) is not int or summary["num_samples"] <= 0:
        raise ValueError("Invalid report num_samples")
    for key in ("score", "tie_break_score"):
        value = summary.get(key)
        if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid report {key}")
    duration = summary.get("average_processing_seconds")
    if duration is not None and (type(duration) not in (int, float) or not isfinite(duration) or duration < 0):
        raise ValueError("Invalid average_processing_seconds")
    return summary


def aggregate_five_runs(reports: list[dict]) -> dict:
    """Average exactly five separately executed runs, never re-run here."""
    if len(reports) != 5:
        raise ValueError("The reproducibility rule requires exactly 5 independently measured runs")
    summaries = [_summary(report) for report in reports]
    if any(s.get("num_runs") != 1 for s in summaries):
        raise ValueError("Provide 5 single-run reports, not pre-averaged reports")
    if any(report.get("verification") != reports[0].get("verification") for report in reports):
        raise ValueError("Five-run reports use different zaic_tools versions")
    for key in ("num_samples", "gold_fingerprint", "catalog_fingerprint", "hash_verification", "timing_available"):
        if any(summary.get(key) != summaries[0].get(key) for summary in summaries):
            raise ValueError(f"Five-run reports disagree on {key}")
    sample_maps = []
    for report in reports:
        sample_map = {}
        for sample in report.get("samples", []):
            if sample.get("id") in sample_map:
                raise ValueError("Duplicate sample id in run report")
            sample_map[sample["id"]] = sample
        if len(sample_map) != summaries[0]["num_samples"]:
            raise ValueError("Missing sample details in run report")
        sample_maps.append(sample_map)
    if any(samples.keys() != sample_maps[0].keys() for samples in sample_maps):
        raise ValueError("Five-run sample IDs differ")
    summary = {
        "metric_version": 2, "score_scale": 100, "num_runs": 5,
        "num_samples": summaries[0]["num_samples"],
        "gold_fingerprint": summaries[0]["gold_fingerprint"],
        "catalog_fingerprint": summaries[0]["catalog_fingerprint"],
        "hash_verification": summaries[0]["hash_verification"],
        "score": fsum(s["score"] for s in summaries) / 5,
        "mean_sample_score": fsum(s["mean_sample_score"] for s in summaries) / 5,
        "tie_break_score": fsum(s["tie_break_score"] for s in summaries) / 5,
        "timing_available": summaries[0]["timing_available"],
        "average_processing_seconds": (
            fsum(s["average_processing_seconds"] for s in summaries) / 5
            if all(s.get("average_processing_seconds") is not None for s in summaries) else None
        ),
    }
    result = {
        "summary": summary,
        "runs": [{"run": i + 1, "summary": s} for i, s in enumerate(summaries)],
        "samples": [{
            "id": key,
            "score": fsum(samples[key]["score"] for samples in sample_maps) / 5,
            "tie_break_f1": fsum(samples[key]["tie_break_f1"] for samples in sample_maps) / 5,
        } for key in sample_maps[0]],
    }
    if reports[0].get("verification") is not None:
        result["verification"] = reports[0]["verification"]
    return result


def rank_reports(entries: list[tuple[str, dict]]) -> list[dict]:
    """Rank by score, F1 ignoring hashes, then lower mean inference time.

    No rounding is applied before comparing scores. Exact equality remains a
    shared rank. A time tie-break requires real timing evidence for both teams.
    """
    names = set()
    groups = defaultdict(list)
    reference = None
    for name, report in entries:
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Ranking entries need distinct, non-empty names")
        names.add(name)
        summary = _summary(report)
        signature = (
            *(summary.get(k) for k in ("gold_fingerprint", "catalog_fingerprint", "hash_verification", "num_runs")),
            report.get("verification"),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Ranking reports must use the same gold, catalog, verification mode and run count")
        groups[(summary["score"], summary["tie_break_score"])].append((name, summary))
    if not entries:
        raise ValueError("Ranking needs at least one entry")
    ordered = []
    for _, group in sorted(groups.items(), reverse=True):
        if len(group) > 1 and any(s["average_processing_seconds"] is None for _, s in group):
            raise ValueError("Time tie-break requires complete --timings for tied entries")
        ordered.extend(sorted(group, key=lambda entry: (entry[1]["average_processing_seconds"] or 0.0, entry[0])))
    output = []
    previous, rank = None, 0
    for position, (name, summary) in enumerate(ordered, 1):
        key = (summary["score"], summary["tie_break_score"], summary["average_processing_seconds"])
        if key != previous:
            rank = position
        previous = key
        output.append({
            "rank": rank, "name": name, "score": summary["score"],
            "tie_break_score": summary["tie_break_score"],
            "average_processing_seconds": summary["average_processing_seconds"],
        })
    return output
