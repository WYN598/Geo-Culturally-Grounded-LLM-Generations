from typing import Dict, List


def mcq_accuracy(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if (r.get("pred", "") == r.get("answer", "")))
    return correct / len(rows)
