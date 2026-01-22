#!/usr/bin/env python3
"""
Compare Base Model vs Cartridge (ICL) vs Transmuted Adapter on LongHealth & MTOB.

Usage:
  python examples/benchmarks/compare_models.py \
    --model-name Qwen/Qwen3-0.6b \
    --adapter-path /path/to/transmuted_adapter.pt \
    --cartridge-ids hazyresearch/m07d11... hazyresearch/m07d28... \
    --device cuda
"""

import argparse
import time
from typing import List, Optional, Union, Dict
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from cartridges.transmutation.adapter import MultiLayerThoughtAdapter
from cartridges.utils.hf import read_conversations_from_hf
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset

class ModelRunner:
    def __init__(self, model_name: str, adapter_paths: Optional[List[str]] = None, device: str = "cuda"):
        print(f"Loading model: {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        self.has_adapter = False
        self.handles = []
        
        if adapter_paths and len(adapter_paths) > 0:
            print(f"Loading and merging adapters from: {adapter_paths}")
            self.adapter = self._load_and_merge_adapters(adapter_paths)
            self._apply_adapter()
            self.has_adapter = True

    def _load_and_merge_adapters(self, paths: List[str]) -> MultiLayerThoughtAdapter:
        merged_adapters = {} # layer_idx -> ThoughtAdapter
        
        for p in paths:
            print(f"  - Loading {p}...")
            # Load individual adapter container
            container = MultiLayerThoughtAdapter.from_pretrained(p, map_location="cpu")
            
            for layer_idx, adapter in container.adapters.items():
                if layer_idx not in merged_adapters:
                    # First time seeing this layer: just copy
                    # Clone tensors to be safe
                    merged_adapters[layer_idx] = adapter
                else:
                    # Layer collision: add deltas
                    # We modify the existing object in merged_adapters
                    existing = merged_adapters[layer_idx]
                    # Create new adapter with summed weights to avoid mutating the original reference if shared
                    # But since we loaded fresh, we can mutate safely or create new.
                    existing.bias_delta = existing.bias_delta + adapter.bias_delta
                    existing.weight_delta = existing.weight_delta + adapter.weight_delta
                    
        return MultiLayerThoughtAdapter(merged_adapters)

    def _apply_adapter(self):
        # Selector for standard HF models
        def layer_selector(m, i):
            if hasattr(m, "model") and hasattr(m.model, "layers"):
                return m.model.layers[i]
            elif hasattr(m, "layers"):
                return m.layers[i]
            raise ValueError(f"Could not locate layer {i} in model {type(m)}")

        self.handles = self.adapter.apply(self.model, layer_selector)

    def generate(self, prompts: List[Union[str, List[Dict]]], max_new_tokens: int = 128, context: str = "") -> List[str]:
        """
        Generate responses.
        prompts: list of strings or list of chat messages.
        context: Optional string to prepend (system prompt / ICL context).
        """
        results = []
        for prompt in tqdm(prompts, desc="Generating"):
            # Prepare input text
            if isinstance(prompt, list):
                # Apply chat template
                # If we have context, we might want to insert it into system prompt
                if context:
                    # Check if system message exists
                    if prompt[0]['role'] == 'system':
                        prompt[0]['content'] = context + "\n\n" + prompt[0]['content']
                    else:
                        prompt.insert(0, {'role': 'system', 'content': context})
                
                text_input = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            else:
                # String prompt
                if context:
                    text_input = f"{context}\n\n{prompt}"
                else:
                    text_input = prompt

            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # Greedy decoding for benchmarks
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_ids = out[0][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(response_text)
            
        return results

    def unload(self):
        # Remove hooks if any
        for h in self.handles:
            h.remove()
        self.handles = []
        # We don't delete model here to reuse it if needed, but for this script we might just re-instantiate or keep it.


def load_cartridges_text(cartridge_ids: List[str]) -> str:
    print(f"Loading cartridges: {cartridge_ids}")
    full_text = []
    for cid in cartridge_ids:
        print(f"Reading {cid}...")
        convos = read_conversations_from_hf(cid)
        # Flatten conversations to text
        for conv in convos:
            for msg in conv.messages:
                if msg.content:
                    full_text.append(msg.content)
    
    # Join with newlines
    combined = "\n\n".join(full_text)
    print(f"Loaded total {len(combined)} chars of context.")
    return combined


def run_longhealth(runner: ModelRunner, context: str = "", limit: int = 16):
    print("\n--- Running LongHealth Benchmark ---")
    dataset = LongHealthMultipleChoiceGenerateDataset(
        LongHealthMultipleChoiceGenerateDataset.Config(max_questions=limit),
        tokenizer=runner.tokenizer,
        seed=42
    )
    
    correct = 0
    total = 0
    
    prompts = [elem.prompt for elem in dataset]
    # We run generation
    responses = runner.generate(prompts, max_new_tokens=64, context=context)
    
    for i, resp in enumerate(responses):
        elem = dataset[i]
        # Use dataset's scoring
        score, _ = dataset.score(resp, elem.answer, elem.convo_id)
        if score:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"LongHealth Accuracy: {acc:.2%} ({correct}/{total})")
    return acc, correct, total


def run_mtob(runner: ModelRunner, context: str = "", limit: int = 16):
    print("\n--- Running MTOB Benchmark ---")
    dataset = MTOBKalamangToEnglishGenerateDataset(
        MTOBKalamangToEnglishGenerateDataset.Config(),
        tokenizer=runner.tokenizer,
        seed=42
    )
    
    correct = 0
    total = 0
    
    # Manually limit here
    all_prompts = [elem.prompt for elem in dataset]
    if limit:
        prompts = all_prompts[:limit]
    else:
        prompts = all_prompts

    # Short generation for translation
    responses = runner.generate(prompts, max_new_tokens=64, context=context)
    
    for i, resp in enumerate(responses):
        elem = dataset[i]
        # Simple exact match (normalized)
        if resp.strip().lower() == elem.answer.strip().lower():
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"MTOB Accuracy: {acc:.2%} ({correct}/{total})")
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    # Changed to accept multiple paths
    parser.add_argument("--adapter-paths", nargs="+", required=True)
    parser.add_argument("--cartridge-ids", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=32, help="Max samples per benchmark")
    args = parser.parse_args()

    # 1. Load Cartridge Text for ICL
    cartridge_context = load_cartridges_text(args.cartridge_ids)
    
    # 2. Results Dictionary
    results = {}

    # --- Condition 1: Base Model ---
    print("\n\n=== Condition 1: Base Model ===")
    runner = ModelRunner(args.model_name, device=args.device)
    lh_acc, lh_correct, lh_total = run_longhealth(runner, limit=args.limit)
    mtob_acc, mtob_correct, mtob_total = run_mtob(runner, limit=args.limit)
    
    # Calculate aggregate accuracy
    base_aggregate_correct = lh_correct + mtob_correct
    base_aggregate_total = lh_total + mtob_total
    base_aggregate_acc = base_aggregate_correct / base_aggregate_total if base_aggregate_total > 0 else 0
    
    results["Base_LongHealth"] = lh_acc
    results["Base_MTOB"] = mtob_acc
    results["Base_Overall"] = base_aggregate_acc
    print(f"Base Overall Accuracy: {base_aggregate_acc:.2%} ({base_aggregate_correct}/{base_aggregate_total})")
    
    # --- Condition 2: Base Model + Cartridges (ICL) ---
    print("\n\n=== Condition 2: Base Model + Concatenated Cartridges (ICL) ===")
    # Reuse runner, just pass context
    # Warning logic remains the same
    if len(cartridge_context) > 100000:
        print("Warning: Context is very large, truncating to last 50000 chars for safety.")
        cartridge_context = cartridge_context[-50000:]
        
    lh_acc, lh_correct, lh_total = run_longhealth(runner, context=cartridge_context, limit=args.limit)
    mtob_acc, mtob_correct, mtob_total = run_mtob(runner, context=cartridge_context, limit=args.limit)
    
    # Calculate aggregate accuracy
    cart_aggregate_correct = lh_correct + mtob_correct
    cart_aggregate_total = lh_total + mtob_total
    cart_aggregate_acc = cart_aggregate_correct / cart_aggregate_total if cart_aggregate_total > 0 else 0
    
    results["Cartridge_LongHealth"] = lh_acc
    results["Cartridge_MTOB"] = mtob_acc
    results["Cartridge_Overall"] = cart_aggregate_acc
    print(f"Cartridge Overall Accuracy: {cart_aggregate_acc:.2%} ({cart_aggregate_correct}/{cart_aggregate_total})")
    
    # Clean up runner
    del runner
    torch.cuda.empty_cache()

    # --- Condition 3: Transmuted Model ---
    print(f"\n\n=== Condition 3: Transmuted Model (Weights from {len(args.adapter_paths)} files) ===")
    runner = ModelRunner(args.model_name, adapter_paths=args.adapter_paths, device=args.device)
    lh_acc, lh_correct, lh_total = run_longhealth(runner, limit=args.limit)
    mtob_acc, mtob_correct, mtob_total = run_mtob(runner, limit=args.limit)
    
    # Calculate aggregate accuracy
    trans_aggregate_correct = lh_correct + mtob_correct
    trans_aggregate_total = lh_total + mtob_total
    trans_aggregate_acc = trans_aggregate_correct / trans_aggregate_total if trans_aggregate_total > 0 else 0
    
    results["Transmuted_LongHealth"] = lh_acc
    results["Transmuted_MTOB"] = mtob_acc
    results["Transmuted_Overall"] = trans_aggregate_acc
    print(f"Transmuted Overall Accuracy: {trans_aggregate_acc:.2%} ({trans_aggregate_correct}/{trans_aggregate_total})")

    # --- Summary ---
    print("\n\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.2%}")

if __name__ == "__main__":
    main()
