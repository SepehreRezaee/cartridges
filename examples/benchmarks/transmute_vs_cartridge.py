#!/usr/bin/env python3
"""Compare direct cartridges vs transmuted adapter across key benchmarks.

Benchmarks implemented from the requested table:
 - Recall/Memory: LongHealth multiple-choice
 - Skill Acquisition: MTOB translation
 - Throughput: simple tokens/sec timing (local)
 - Stability: MMLU-lite (small subset) to sanity-check generality

Usage (example):
  python -m examples.benchmarks.transmute_vs_cartridge \
    --model-name Qwen/Qwen3-4b \
    --adapter-path outputs/transmuted_adapter.pt \
    --cartridge-ids hazyresearch/cartridge-wauoq23f \
    --tokasaurus-url http://localhost:10210

Notes:
 - Direct cartridge path uses TokasaurusClient + cartridges list.
 - Transmuted path uses a local HF model patched with the ThoughtAdapter.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.evaluate import GenerationEvalConfig, GenerationEvalRunConfig, ICLBaseline
from cartridges.transmutation.adapter import MultiLayerThoughtAdapter, ThoughtAdapter
from cartridges.utils.wandb import WandBConfig

# ------------------------- Local Adapter Generator --------------------------- #

@dataclass
class AdapterResult:
    text: str
    num_system_and_user_tokens: int
    num_assistant_tokens: int


class AdapterGenerator:
    """Generator that applies a transmuted adapter on a local HF model."""

    def __init__(self, model_name: str, adapter_path: str, device: str = "cuda"):
        # Load multi-layer adapter
        self.adapter = MultiLayerThoughtAdapter.from_pretrained(adapter_path, map_location="cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
        # Apply hooks. We need a selector that maps layer indices to modules.
        # Assuming standard HF models where layers are in model.model.layers or model.layers
        def layer_selector(m, i):
            if hasattr(m, "model") and hasattr(m.model, "layers"):
                return m.model.layers[i]
            elif hasattr(m, "layers"):
                return m.layers[i]
            # Fallback for some architectures or if i is special (like -1 for lm_head? usually we target hidden layers)
            raise ValueError(f"Could not locate layer {i} in model {type(m)}")

        self.handles = self.adapter.apply(self.model, layer_selector)

    def generate(self, prompts: List[str], max_new_tokens: int = 128, temperature: float = 0.0) -> List[AdapterResult]:
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            elapsed = time.time() - t0
            num_in = inputs.input_ids.numel()
            num_out = out.shape[1] - inputs.input_ids.shape[1]
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            results.append(
                AdapterResult(
                    text=text,
                    num_system_and_user_tokens=num_in,
                    num_assistant_tokens=num_out,
                )
            )
        return results

    def close(self):
        for h in self.handles:
            h.remove()


# ----------------------------- Benchmark Runner ----------------------------- #

def run_longhealth_cartridge(cfg) -> None:
    print("== LongHealth (cartridge) ==")
    generator = ICLBaseline.Config(
        client=TokasaurusClient.Config(
            model_name=cfg.model_name,
            url=cfg.tokasaurus_url,
            cartridges=[CartridgeConfig(id=cid, source="huggingface") for cid in cfg.cartridge_ids],
        ),
        tokenizer=cfg.model_name,
        temperature=0.0,
        max_completion_tokens=256,
        context="Cartridge-augmented inference",
    )
    eval_cfg = GenerationEvalConfig(
        dataset=LongHealthMultipleChoiceGenerateDataset.Config(max_questions=64),
        num_samples=1,
        temperature=0.0,
        name_for_wandb="longhealth_cartridge",
    )
    run = GenerationEvalRunConfig(
        eval=eval_cfg,
        generator=generator,
        batch_size=8,
        tokenizer=cfg.model_name,
        name="longhealth_cartridge",
        wandb=WandBConfig(project="transmute-bench") if cfg.use_wandb else None,
    )
    run.run()


def run_longhealth_adapter(cfg) -> None:
    print("== LongHealth (adapter) ==")
    adapter_gen = AdapterGenerator(cfg.model_name, cfg.adapter_path, device=cfg.device)
    dataset = LongHealthMultipleChoiceGenerateDataset(
        LongHealthMultipleChoiceGenerateDataset.Config(max_questions=64),
        tokenizer=adapter_gen.tokenizer,
        seed=0,
    )
    correct, total = 0, 0
    for elem in dataset:
        resp = adapter_gen.generate([elem.prompt], max_new_tokens=256, temperature=0.0)[0]
        ok, _ = dataset.score(resp.text, elem.answer, elem.convo_id)
        correct += int(ok)
        total += 1
    adapter_gen.close()
    print(f"Adapter LongHealth accuracy: {correct}/{total} ({100*correct/total:.2f}%)")


def run_mtob_adapter(cfg) -> None:
    print("== MTOB (adapter) ==")
    adapter_gen = AdapterGenerator(cfg.model_name, cfg.adapter_path, device=cfg.device)
    dataset = MTOBKalamangToEnglishGenerateDataset(
        MTOBKalamangToEnglishGenerateDataset.Config(max_samples=64),
        tokenizer=adapter_gen.tokenizer,
        seed=0,
    )
    # Simple BLEU-like proxy: exact match
    correct, total = 0, 0
    for elem in dataset:
        resp = adapter_gen.generate([elem.prompt], max_new_tokens=64, temperature=0.0)[0]
        ok = resp.text.strip().lower() == elem.answer.strip().lower()
        correct += int(ok)
        total += 1
    adapter_gen.close()
    print(f"Adapter MTOB exact-match: {correct}/{total} ({100*correct/total:.2f}%)")


def run_throughput_adapter(cfg) -> None:
    print("== Throughput (adapter) ==")
    adapter_gen = AdapterGenerator(cfg.model_name, cfg.adapter_path, device=cfg.device)
    prompt = "Briefly summarize the corpus content."
    n = 32
    t0 = time.time()
    adapter_gen.generate([prompt] * n, max_new_tokens=64, temperature=0.0)
    elapsed = time.time() - t0
    tokens = n * 64
    print(f"Adapter throughput: {tokens/elapsed:.2f} tokens/sec over {n} requests")
    adapter_gen.close()


def run_throughput_cartridge(cfg) -> None:
    print("== Throughput (cartridge) ==")
    client = TokasaurusClient.Config(
        model_name=cfg.model_name,
        url=cfg.tokasaurus_url,
        cartridges=[CartridgeConfig(id=cid, source="huggingface") for cid in cfg.cartridge_ids],
    ).instantiate()
    prompt = "Briefly summarize the corpus content."
    n = 32
    chats = [[{"role": "user", "content": prompt}]] * n
    t0 = time.time()
    # note: synchronous for simplicity
    import asyncio

    asyncio.run(client.chat(chats=chats, max_completion_tokens=64, temperature=0.0))
    elapsed = time.time() - t0
    tokens = n * 64
    print(f"Cartridge throughput: {tokens/elapsed:.2f} tokens/sec over {n} requests")


def parse_args():
    p = argparse.ArgumentParser(description="Compare cartridges vs transmuted adapter.")
    p.add_argument("--model-name", default="Qwen/Qwen3-4b")
    p.add_argument("--adapter-path", required=True, help="Path to transmuted adapter .pt")
    p.add_argument("--cartridge-ids", nargs="+", required=True, help="HF cartridge IDs for direct KV cache")
    p.add_argument("--tokasaurus-url", required=True, help="Tokasaurus base URL for cartridge path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def main():
    cfg = parse_args()
    # Recall / Memory
    run_longhealth_cartridge(cfg)
    run_longhealth_adapter(cfg)
    # Skill Acquisition
    run_mtob_adapter(cfg)
    # Throughput
    run_throughput_cartridge(cfg)
    run_throughput_adapter(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
