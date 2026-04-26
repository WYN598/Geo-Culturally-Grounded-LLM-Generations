import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def benchmark_configs(limit: int) -> List[Tuple[str, str]]:
    suffix = str(limit) if limit > 0 else "full"
    return [
        (f"blend_{suffix}", "configs/legacy_single/config_blend_200.yaml"),
        (f"normad_{suffix}", "configs/legacy_single/config_normad_200.yaml"),
        (f"seegull_{suffix}", "configs/legacy_single/config_seegull_200.yaml"),
        (f"bbq_{suffix}", "configs/external/config_bbq_200.yaml"),
        (f"cbbq_{suffix}", "configs/external/config_cbbq_200.yaml"),
        (f"borderlines_{suffix}", "configs/external/config_borderlines_200.yaml"),
        (f"msqad_{suffix}", "configs/external/config_msqad_200.yaml"),
        (f"socialstigmaqa_{suffix}", "configs/external/config_socialstigmaqa_200.yaml"),
        (f"truthfulqa_{suffix}", "configs/external/config_truthfulqa_200.yaml"),
        (f"popqa_{suffix}", "configs/external/config_popqa_200.yaml"),
        (f"honest_{suffix}", "configs/external/config_honest_200.yaml"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five-way benchmark ablations without full general RAG.")
    parser.add_argument("--out-root", default="outputs/fiveway_benchmark_suite")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_cmd([sys.executable, "scripts/build_legacy_single_configs.py"])

    for tag, config_path in benchmark_configs(args.limit):
        cmd = [
            sys.executable,
            "scripts/run_external_ablation.py",
            "--config",
            config_path,
            "--out-root",
            args.out_root,
            "--tag",
            tag,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--no-full-general",
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.refresh_cache:
            cmd.append("--refresh-cache")
        run_cmd(cmd)


if __name__ == "__main__":
    main()
