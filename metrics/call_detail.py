"""Diagnostic breakdown of tool calls: tool selection vs argument accuracy.

Not part of the official score. For each sample, predicted and gold calls are
compared in two steps:

* tool selection: multiset overlap of tool names (precision / recall / F1);
* argument accuracy: calls with the same tool name are paired (greedy, most
  agreeing arguments first) and each argument is classified as ``match``,
  ``wrong``, ``missing`` (in gold only) or ``extra`` (in prediction only),
  grouped by argument kind (id, enum, bool, number, datetime, fuzzy, text).

Argument equality reuses the official rule (``x-match: fuzzy`` → token-F1 > 0.6).
"""

from collections import Counter, defaultdict
from typing import Iterable, Mapping
import re

from .scorer import ToolCatalog, _arguments_match, _valid_call

_DATETIME = re.compile(r"YYYY|HH:MM|ISO", re.I)
KINDS = ("id", "enum", "bool", "number", "datetime", "fuzzy", "text", "other")
SAMPLE_CLASSES = ("exact", "tool_selection", "arguments", "both")


def argument_kind(key: str, schema: Mapping) -> str:
    if schema.get("x-match") == "fuzzy":
        return "fuzzy"
    if "enum" in schema:
        return "enum"
    kind = schema.get("type")
    if kind == "boolean":
        return "bool"
    if kind in ("integer", "number"):
        return "number"
    if key.endswith("_id") or key.endswith("_ids") or key == "id":
        return "id"
    if kind == "string":
        return "datetime" if _DATETIME.search(schema.get("description", "")) or schema.get("format") else "text"
    return "other"


def _pairs(predicted: list, gold: list, catalog: ToolCatalog) -> list[tuple[dict, dict]]:
    """Pair same-name calls, preferring pairs with more agreeing arguments."""
    by_name = defaultdict(lambda: ([], []))
    for call in predicted:
        by_name[call["name"]][0].append(call)
    for call in gold:
        by_name[call["name"]][1].append(call)
    pairs = []
    for name, (left, right) in by_name.items():
        props = catalog.parameters.get(name, {}).get("properties", {})
        scored = sorted(
            ((sum(k in g["arguments"] and _arguments_match(v, g["arguments"][k], props.get(k, {}))
                  for k, v in p["arguments"].items()) - 0.01 * len(set(p["arguments"]) ^ set(g["arguments"])), i, j)
             for i, p in enumerate(left) for j, g in enumerate(right)),
            reverse=True,
        )
        used_left, used_right = set(), set()
        for _, i, j in scored:
            if i not in used_left and j not in used_right:
                used_left.add(i)
                used_right.add(j)
                pairs.append((left[i], right[j]))
    return pairs


def call_breakdown(submitted: Iterable[Mapping], gold: Iterable[Mapping], catalog: ToolCatalog) -> dict:
    gold_by_id = {row["id"]: row for row in gold}
    totals = Counter()
    by_kind = {kind: Counter() for kind in KINDS}
    by_argument = defaultdict(Counter)
    by_tool = defaultdict(Counter)
    classes = Counter()
    samples = []
    for row in submitted:
        reference = gold_by_id.get(row["id"])
        if reference is None:
            continue
        predicted = [c for c in row.get("tool_calls") or [] if _valid_call(c)]
        expected = [c for c in reference.get("tool_calls") or [] if _valid_call(c)]
        names_p, names_g = Counter(c["name"] for c in predicted), Counter(c["name"] for c in expected)
        overlap = sum((names_p & names_g).values())
        totals.update(samples=1, predicted_calls=len(predicted), gold_calls=len(expected), name_matched=overlap)
        for name, count in (names_g - names_p).items():
            by_tool[name]["missing"] += count
        for name, count in (names_p - names_g).items():
            by_tool[name]["extra"] += count
        wrong_arguments = 0
        for p, g in _pairs(predicted, expected, catalog):
            name = p["name"]
            props = catalog.parameters.get(name, {}).get("properties", {})
            exact = catalog.calls_match(p, g)
            totals.update(paired_calls=1, paired_exact=int(exact))
            by_tool[name]["paired"] += 1
            by_tool[name]["paired_exact"] += int(exact)
            for key in sorted(set(p["arguments"]) | set(g["arguments"])):
                schema = props.get(key, {})
                if key not in p["arguments"]:
                    status = "missing"
                elif key not in g["arguments"]:
                    status = "extra"
                else:
                    status = "match" if _arguments_match(p["arguments"][key], g["arguments"][key], schema) else "wrong"
                kind = argument_kind(key, schema)
                by_kind[kind][status] += 1
                by_argument[f"{name}.{key}"][status] += 1
                wrong_arguments += status != "match"
        selection_ok = names_p == names_g
        sample_class = ("exact" if not wrong_arguments else "arguments") if selection_ok else (
            "both" if wrong_arguments else "tool_selection")
        classes[sample_class] += 1
        samples.append({"id": row["id"], "class": sample_class, "missing_tools": sorted((names_g - names_p).elements()),
                        "extra_tools": sorted((names_p - names_g).elements()), "wrong_arguments": wrong_arguments})

    def rate(counter: Counter) -> dict:
        total = sum(counter.values())
        return {**{k: counter[k] for k in ("match", "wrong", "missing", "extra")}, "total": total,
                "accuracy": counter["match"] / total if total else None}

    p_total, g_total, matched = totals["predicted_calls"], totals["gold_calls"], totals["name_matched"]
    summary = {
        "num_samples": totals["samples"],
        "tool_selection": {
            "matched": matched, "predicted_calls": p_total, "gold_calls": g_total,
            "precision": matched / p_total if p_total else None,
            "recall": matched / g_total if g_total else None,
            "f1": 2 * matched / (p_total + g_total) if p_total + g_total else None,
        },
        "argument_accuracy": {
            "paired_calls": totals["paired_calls"], "paired_exact": totals["paired_exact"],
            "call_exact_rate": totals["paired_exact"] / totals["paired_calls"] if totals["paired_calls"] else None,
            "by_kind": {kind: rate(counter) for kind, counter in by_kind.items() if counter},
        },
        "sample_classes": {name: classes[name] for name in SAMPLE_CLASSES},
    }
    worst_arguments = sorted(
        ({"argument": key, **rate(counter)} for key, counter in by_argument.items()),
        key=lambda item: (item["total"] - item["match"], item["total"]), reverse=True,
    )
    worst_tools = sorted(
        ({"tool": name, **{k: counter[k] for k in ("missing", "extra", "paired", "paired_exact")}}
         for name, counter in by_tool.items()),
        key=lambda item: item["missing"] + item["extra"] + item["paired"] - item["paired_exact"], reverse=True,
    )
    return {"summary": summary, "arguments": worst_arguments, "tools": worst_tools, "samples": samples}
