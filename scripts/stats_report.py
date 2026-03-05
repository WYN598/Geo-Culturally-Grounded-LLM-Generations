import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_acc(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if str(r.get("pred", "")) == str(r.get("answer", "")))
    return correct / len(rows)


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    x = sorted(vals)
    idx = int((len(x) - 1) * p)
    return x[idx]


def bootstrap_ci(values: List[float], n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    # Numerically-stable Binomial(n, 0.5) lower-tail sum in log-space.
    logs = []
    for i in range(0, k + 1):
        lp = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - n * math.log(2.0)
        )
        logs.append(lp)

    m = max(logs)
    cdf = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2.0 * cdf)


def permutation_paired(values: List[float], n_perm: int = 10000, seed: int = 42) -> float:
    if not values:
        return 1.0
    rng = random.Random(seed)
    obs = abs(sum(values) / len(values))
    exceed = 0
    for _ in range(n_perm):
        sim = []
        for v in values:
            if v == 0:
                sim.append(0.0)
            else:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                sim.append(abs(v) * sign)
        stat = abs(sum(sim) / len(sim))
        if stat >= obs:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def align_pairs(a_rows: List[Dict], b_rows: List[Dict]) -> List[Tuple[Dict, Dict]]:
    a_map = {str(r.get("id", "")): r for r in a_rows}
    b_map = {str(r.get("id", "")): r for r in b_rows}
    ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    return [(a_map[i], b_map[i]) for i in ids if i]


def _normalized_choices(row: Dict) -> List[str]:
    out = []
    for c in row.get("choices", []) or []:
        s = str(c).strip()
        if ")" in s:
            s = s.split(")", 1)[-1].strip()
        out.append(s.lower())
    return out


def _stereotype_pred(row: Dict) -> Optional[float]:
    # Only define stereotype-rate if option A is explicitly "stereotype".
    choices = _normalized_choices(row)
    if not choices:
        return None
    first = choices[0]
    if "stereotype" in first and "non-stereotype" not in first:
        return 1.0 if str(row.get("pred", "")).upper() == "A" else 0.0
    return None


def analyze_pair(base_rows: List[Dict], test_rows: List[Dict], n_boot: int, n_perm: int) -> Dict:
    pairs = align_pairs(base_rows, test_rows)

    base_correct = []
    test_correct = []
    diff_correct = []

    for b, t in pairs:
        bc = 1.0 if str(b.get("pred", "")) == str(b.get("answer", "")) else 0.0
        tc = 1.0 if str(t.get("pred", "")) == str(t.get("answer", "")) else 0.0
        base_correct.append(bc)
        test_correct.append(tc)
        diff_correct.append(tc - bc)

    b_only = sum(1 for bc, tc in zip(base_correct, test_correct) if bc == 1.0 and tc == 0.0)
    t_only = sum(1 for bc, tc in zip(base_correct, test_correct) if bc == 0.0 and tc == 1.0)

    overall = {
        "n": len(pairs),
        "base_acc": (sum(base_correct) / len(base_correct)) if base_correct else 0.0,
        "test_acc": (sum(test_correct) / len(test_correct)) if test_correct else 0.0,
        "acc_diff_test_minus_base": (sum(diff_correct) / len(diff_correct)) if diff_correct else 0.0,
        "acc_diff_bootstrap_ci95": bootstrap_ci(diff_correct, n_boot=n_boot),
        "mcnemar": {
            "base_only_correct": b_only,
            "test_only_correct": t_only,
            "p_value_exact": mcnemar_exact_p(b_only, t_only),
        },
    }

    ds_names = sorted({str(p[0].get("dataset", "unknown")) for p in pairs})
    by_dataset = {}
    for ds in ds_names:
        sub_base = [p[0] for p in pairs if str(p[0].get("dataset", "unknown")) == ds]
        sub_test = [p[1] for p in pairs if str(p[0].get("dataset", "unknown")) == ds]
        sub_pairs = list(zip(sub_base, sub_test))
        dvals = []
        for b, t in sub_pairs:
            bc = 1.0 if str(b.get("pred", "")) == str(b.get("answer", "")) else 0.0
            tc = 1.0 if str(t.get("pred", "")) == str(t.get("answer", "")) else 0.0
            dvals.append(tc - bc)
        by_dataset[ds] = {
            "n": len(sub_pairs),
            "base_acc": group_acc(sub_base),
            "test_acc": group_acc(sub_test),
            "acc_diff_test_minus_base": (sum(dvals) / len(dvals)) if dvals else 0.0,
            "acc_diff_bootstrap_ci95": bootstrap_ci(dvals, n_boot=n_boot),
        }

    seegull_pairs = [p for p in pairs if str(p[0].get("dataset", "")).lower() == "seegull"]
    stereo = {"n": 0}
    if seegull_pairs:
        base_vals = []
        test_vals = []
        diffs = []
        for b, t in seegull_pairs:
            bs = _stereotype_pred(b)
            ts = _stereotype_pred(t)
            if bs is None or ts is None:
                continue
            base_vals.append(bs)
            test_vals.append(ts)
            diffs.append(ts - bs)

        if diffs:
            n = len(diffs)
            stereo = {
                "n": n,
                "base_stereotype_rate": sum(base_vals) / n,
                "test_stereotype_rate": sum(test_vals) / n,
                "stereotype_rate_diff_test_minus_base": sum(diffs) / n,
                "stereotype_diff_bootstrap_ci95": bootstrap_ci(diffs, n_boot=n_boot),
                "paired_permutation_p": permutation_paired(diffs, n_perm=n_perm),
            }
        else:
            stereo = {
                "n": 0,
                "note": "Stereotype-rate is undefined for this SeeGULL task variant (option A is not explicit stereotype).",
            }

    return {
        "overall": overall,
        "by_dataset": by_dataset,
        "seegull_stereotype": stereo,
    }


def run(args):
    base_rows = load_jsonl(args.base)
    selective_rows = load_jsonl(args.search_selective)

    report = {
        "comparison_search_selective_vs_base": analyze_pair(
            base_rows, selective_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )
    }

    if args.search_non_selective:
        non_rows = load_jsonl(args.search_non_selective)
        report["comparison_search_non_selective_vs_base"] = analyze_pair(
            base_rows, non_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )
        report["comparison_search_selective_vs_non_selective"] = analyze_pair(
            non_rows, selective_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out": args.out, **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical report for strict search-grounding experiments.")
    parser.add_argument("--base", required=True, help="Baseline predictions jsonl (e.g., vanilla_predictions.jsonl)")
    parser.add_argument("--search-selective", required=True, help="Selective search predictions jsonl")
    parser.add_argument("--search-non-selective", default="", help="Optional non-selective search predictions jsonl")
    parser.add_argument("--out", default="outputs/strict_search/stats_report.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    run(args)
