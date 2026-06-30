#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRETRAINED_MODEL_IDS = [
    "BAAI/bge-m3",
    "kietnt0603/nrk-legal-bge",
    "AITeamVN/Vietnamese_Embedding",
]


def main():
    parser = argparse.ArgumentParser(description="Inference-only dense reranker placeholder.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-name",
        default="AITeamVN/Vietnamese_Embedding",
        help="Pretrained/public Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    args = parser.parse_args()

    # No training/fine-tuning happens in this script.
    # The full dense inference implementation is in notebooks/r2ai-bi-encoder-threshold.ipynb.
    # This CLI preserves the reproducible entry point used by scripts/run_pipeline.sh.
    if args.model_name not in PRETRAINED_MODEL_IDS and not Path(args.model_name).exists():
        print(f"Warning: '{args.model_name}' is not one of the documented public model IDs. "
              "It will be treated as a custom/local checkpoint path if available.")

    data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dense reranker CLI placeholder wrote input unchanged.")
    print("Training mode: false. No model weights are updated.")
    print(f"Selected pretrained model: {args.model_name}")
    print("For the submitted experiment, run notebooks/r2ai-bi-encoder-threshold.ipynb.")


if __name__ == "__main__":
    main()
