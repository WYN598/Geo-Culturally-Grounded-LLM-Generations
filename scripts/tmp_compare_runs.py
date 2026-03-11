import glob
import json


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def acc(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r.get("pred", "")) == str(r.get("answer", ""))) / len(rows)


base = "outputs/optimization_round1_clean_stable/vanilla_clean_stable.jsonl"
vrows = load(base)
vmap = {str(r.get("id", "")): r for r in vrows}
datasets = ["BLEnD", "NormAd", "SeeGULL"]

paths = sorted(glob.glob("outputs/optimization_round2_search_layer/search_selective_searchlayer_opt_ddgs*/search_predictions.jsonl"))
for path in paths:
    rows = load(path)
    smap = {str(r.get("id", "")): r for r in rows}
    ids = sorted(set(vmap.keys()) & set(smap.keys()))
    print("\n" + path)
    merged = [smap[i] for i in ids]
    print("overall", round(acc(merged), 4), "n", len(merged))
    for ds in datasets:
        sub_ids = [i for i in ids if str(vmap[i].get("dataset", "")) == ds]
        vsub = [vmap[i] for i in sub_ids]
        ssub = [smap[i] for i in sub_ids]
        va = acc(vsub)
        sa = acc(ssub)
        print(ds, "van", round(va, 4), "search", round(sa, 4), "diff", round(sa - va, 4))
