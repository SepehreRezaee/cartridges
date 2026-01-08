#!/usr/bin/env python3
"""Concatenate multiple cartridge corpora and transmute them into weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

from cartridges.datasets import DataSource, TrainDataset
from cartridges.datasets import DataSource, TrainDataset
from cartridges.transmutation.adapter import ThoughtAdapter, MultiLayerThoughtAdapter
from cartridges.transmutation.extractor import TokenPatchExtractor
from cartridges.transmutation.pipeline import Transmuter, TransmutationArtifacts
from cartridges.transmutation.solver import ThoughtPatchSolver
from cartridges.utils import get_logger


@dataclass
class CartridgeTransmutationConfig:
    data_paths: List[str]
    model_name: str
    tokenizer_name: Optional[str] = None
    device: str = "cuda"
    packed_seq_length: int = 2048
    seed: int = 0
    lambda_scale: Optional[float] = None
    output_path: str = "transmuted_adapter.pt"
    data_limit: Optional[int] = None
    max_batches: Optional[int] = None
    lambda_scale: Optional[float] = None
    output_path: str = "transmuted_adapter.pt"
    data_limit: Optional[int] = None
    max_batches: Optional[int] = None
    progress: bool = True
    layers: Optional[List[int]] = None  # None means all layers if specified as such


def parse_args() -> CartridgeTransmutationConfig:
    parser = argparse.ArgumentParser(description="Concatenate cartridge corpora and transmute into weights.")
    parser.add_argument(
        "--data-paths",
        nargs="+",
        required=True,
        help="Two or more parquet/pkl files (e.g., two cartridges' self-study datasets).",
    )
    parser.add_argument("--model-name", required=True, help="HF model id or local path (e.g., Qwen/Qwen3-4b).")
    parser.add_argument("--tokenizer-name", default=None, help="Optional separate tokenizer id/path.")
    parser.add_argument("--device", default="cuda", help="Device to run extraction on.")
    parser.add_argument("--packed-seq-length", type=int, default=2048, help="Max packed length for TrainDataset.")
    parser.add_argument("--lambda-scale", type=float, default=None, help="Optional manual lambda in Eq. 24.")
    parser.add_argument("--output-path", default="transmuted_adapter.pt", help="Where to write the adapter weights.")
    parser.add_argument("--data-limit", type=int, default=None, help="Optional cap on number of conversations per file.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on batches for quick debug.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used by TrainDataset.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars.")
    parser.add_argument(
        "--layers", 
        nargs="+", 
        default=["-1"], 
        help="Layers to extract from. Use 'all' for all layers, or list indices (e.g. -1 or 0 1 2)."
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    
    # Parse layers
    if "all" in args.layers:
        layers = None
    else:
        try:
            layers = [int(l) for l in args.layers]
        except ValueError:
            raise ValueError(f"Invalid layers argument: {args.layers}")

    return CartridgeTransmutationConfig(
        data_paths=args.data_paths,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        packed_seq_length=args.packed_seq_length,
        lambda_scale=args.lambda_scale,
        output_path=args.output_path,
        data_limit=args.data_limit,
        max_batches=args.max_batches,
        seed=args.seed,
        progress=args.progress,
        layers=layers,
    )


def build_dataset(cfg: CartridgeTransmutationConfig, tokenizer) -> TrainDataset:
    sources = []
    for p in cfg.data_paths:
        # Auto-detect: if path contains "/" but no "." extension, assume HuggingFace
        if "/" in p and not any(p.endswith(ext) for ext in [".parquet", ".pkl", ".json", ".jsonl"]):
            sources.append(DataSource(path=p, type="hf", limit=cfg.data_limit))
        else:
            sources.append(DataSource(path=p, type="local", limit=cfg.data_limit))
    return TrainDataset(
        TrainDataset.Config(
            data_sources=sources,
            packed_seq_length=cfg.packed_seq_length,
            targets="tokens",
        ),
        tokenizer=tokenizer,
        seed=cfg.seed,
    )


def save_artifacts(artifacts: TransmutationArtifacts, cfg: CartridgeTransmutationConfig) -> None:
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Save dicts for multi-layer
            "bias_deltas": artifacts.bias_deltas,
            "weight_deltas": artifacts.weight_deltas,
            "metadata": artifacts.metadata,
            "model_name": cfg.model_name,
            "tokenizer_name": cfg.tokenizer_name or cfg.model_name,
        },
        out_path,
    )
    print(f"[transmute-cartridges] saved adapter to {out_path}")


def main(cfg: CartridgeTransmutationConfig) -> None:
    logger = get_logger("transmute_cartridges")
    logger.info(f"Loading tokenizer/model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()

    logger.info(f"Building dataset from {len(cfg.data_paths)} sources")
    dataset = build_dataset(cfg, tokenizer)
    logger.info(f"Dataset prepared with {len(dataset.elements)} elements and {len(dataset)} batches")

    logger.info("Starting token patch extraction")
    # Pass cfg.layers
    extractor = TokenPatchExtractor(model=model, tokenizer=tokenizer, layers=cfg.layers, device=cfg.device)
    solver = ThoughtPatchSolver(lambda_scale=cfg.lambda_scale)
    transmuter = Transmuter(extractor=extractor, solver=solver)

    artifacts = transmuter.run(
        dataset=dataset,
        max_batches=cfg.max_batches,
        extra_metadata={"data_paths": cfg.data_paths},
        show_progress=cfg.progress,
    )
    
    # artifacts now has dictionaries of deltas
    num_layers = len(artifacts.bias_deltas)
    logger.info(
        f"Extracted adapter for {num_layers} layers. Lambda={cfg.lambda_scale}"
    )
    save_artifacts(artifacts, cfg)

    # Optional: show how to build an adapter object for reuse.
    # Reconstruct adapter from dictionaries
    adapters = {}
    for idx in artifacts.bias_deltas:
        adapters[idx] = ThoughtAdapter(
            bias_delta=artifacts.bias_deltas[idx],
            weight_delta=artifacts.weight_deltas[idx]
        )
    adapter = MultiLayerThoughtAdapter(adapters)
    
    logger.info("[transmute-cartridges] adapter ready for application via MultiLayerThoughtAdapter.apply()")


if __name__ == "__main__":
    main(parse_args())
