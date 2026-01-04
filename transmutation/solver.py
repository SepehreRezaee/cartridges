from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from tqdm.auto import tqdm

from transmutation.extractor import TokenPatch


@dataclass
class ThoughtPatch:
    """Aggregated "thought" update consisting of bias and weight deltas."""

    bias_delta: torch.Tensor
    weight_delta: torch.Tensor


class ThoughtPatchSolver:
    """Aggregate token-level deltas into a single low-rank update.

    References:
        Eq. 8 (bias) and Eq. 24 (low-rank weight) from "Transmuting Prompts into Weights".
    """

    def __init__(self, lambda_scale: Optional[float] = None) -> None:
        self.lambda_scale = lambda_scale

    def solve(self, patches: Iterable[TokenPatch], show_progress: bool = False) -> dict[int, ThoughtPatch]:
        layer_deltas = {}  # layer_idx -> list of deltas
        layer_outer_sums = {}  # layer_idx -> running sum of outer products
        layer_counts = {}  # layer_idx -> count
        
        # We need to peek at the iterator to know total if show_progress=True, but patches is Iterable
        # So we just use the provided length guess if available or just update without total
        progress = None
        if show_progress:
            try:
                total = len(patches)  # type: ignore[arg-type]
            except TypeError:
                total = None
            progress = tqdm(total=total, desc="Aggregating thought patches", unit="patch")

        for patch in patches:
            l_idx = patch.layer_idx
            delta = patch.delta
            a_t = patch.baseline
            
            if l_idx not in layer_deltas:
                layer_deltas[l_idx] = []
                layer_outer_sums[l_idx] = None
                layer_counts[l_idx] = 0

            layer_deltas[l_idx].append(delta)

            denom = torch.dot(a_t, a_t) + 1e-8
            rank_one = torch.outer(delta, a_t) / denom
            
            if layer_outer_sums[l_idx] is None:
                layer_outer_sums[l_idx] = rank_one
            else:
                layer_outer_sums[l_idx] += rank_one
                
            layer_counts[l_idx] += 1
            
            if progress:
                progress.update(1)

        if progress:
            progress.close()

        if not layer_counts:
            raise ValueError("No patches provided to solver.")

        results = {}
        for l_idx, count in layer_counts.items():
            delta_bias = torch.stack(layer_deltas[l_idx], dim=0).mean(dim=0)
            
            scale = self.lambda_scale if self.lambda_scale is not None else 1.0 / count
            weight_delta = layer_outer_sums[l_idx] * scale
            
            results[l_idx] = ThoughtPatch(bias_delta=delta_bias, weight_delta=weight_delta)

        return results
