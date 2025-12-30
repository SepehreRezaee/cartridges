#!/usr/bin/env python3
"""Run the prompt-to-weights transmutation pipeline on synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydrantic import ObjectConfig

from cartridges.datasets import DataSource, TrainDataset
from cartridges.transmutation.extractor import TokenPatchExtractor
from cartridges.transmutation.solver import ThoughtPatchSolver
from cartridges.utils import get_logger


class TransmutationConfig(ObjectConfig):
    _pass_as_config = True

    data_path: str
    model_name: str
    tokenizer_name: Optional[str] = None
    device: str = "cuda"
    packed_seq_length: int = 2048
    seed: int = 0
    lambda_scale: Optional[float] = None
    output_path: str = "transmuted_adapter.pt"
    data_limit: Optional[int] = None
    progress: bool = True


def parse_args() -> TransmutationConfig:
    parser = argparse.ArgumentParser(description="Transmute prompts into weights.")
    parser.add_argument("--data-path", required=True, help="Path to synthetic parquet/pkl produced by self-study.")
    parser.add_argument("--model-name", required=True, help="HF model id or local path (e.g., Qwen/Qwen3-4b).")
    parser.add_argument("--tokenizer-name", default=None, help="Optional separate tokenizer id/path.")
    parser.add_argument("--device", default="cuda", help="Device to run extraction on.")
    parser.add_argument("--packed-seq-length", type=int, default=2048, help="Max packed length for TrainDataset.")
    parser.add_argument("--lambda-scale", type=float, default=None, help="Optional manual lambda in Eq. 24.")
    parser.add_argument("--output-path", default="transmuted_adapter.pt", help="Where to write the adapter weights.")
    parser.add_argument("--data-limit", type=int, default=None, help="Optional cap on number of conversations.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used by TrainDataset.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars.")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    return TransmutationConfig(
        data_path=args.data_path,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        packed_seq_length=args.packed_seq_length,
        lambda_scale=args.lambda_scale,
        output_path=args.output_path,
        data_limit=args.data_limit,
        seed=args.seed,
        progress=args.progress,
    )


def main(cfg: TransmutationConfig) -> None:
    logger = get_logger("transmute_corpus")
    logger.info(f"Loading tokenizer/model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()

    logger.info(f"Building dataset from {cfg.data_path}")
    ds = TrainDataset(
        TrainDataset.Config(
            data_sources=[DataSource(path=cfg.data_path, type="local", limit=cfg.data_limit)],
            packed_seq_length=cfg.packed_seq_length,
            targets="tokens",
        ),
        tokenizer=tokenizer,
        seed=cfg.seed,
    )
    logger.info(f"Dataset prepared with {len(ds.elements)} elements and {len(ds)} batches")

    logger.info("Starting token patch extraction")
    extractor = TokenPatchExtractor(model=model, tokenizer=tokenizer, device=cfg.device)
    patches = extractor.extract(ds, show_progress=cfg.progress)
    logger.info(f"Extracted {len(patches)} token patches")

    logger.info("Solving for thought patch (weight/bias deltas)")
    solver = ThoughtPatchSolver(lambda_scale=cfg.lambda_scale)
    thought = solver.solve(patches, show_progress=cfg.progress)
    logger.info(
        f"Solved deltas: bias_dim={thought.bias_delta.shape}, "
        f"weight_shape={thought.weight_delta.shape}, lambda={cfg.lambda_scale}"
    )

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bias_delta": thought.bias_delta,
            "weight_delta": thought.weight_delta,
            "model_name": cfg.model_name,
            "tokenizer_name": cfg.tokenizer_name or cfg.model_name,
            "lambda_scale": cfg.lambda_scale,
        },
        out_path,
    )
    logger.info(f"[transmutation] saved adapter to {out_path}")


if __name__ == "__main__":
    main(parse_args())
