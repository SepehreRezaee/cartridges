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

    def solve(self, patches: Iterable[TokenPatch], show_progress: bool = False) -> ThoughtPatch:
        deltas = []
        outer_sums = None
        count = 0
        progress = None

        if show_progress:
            try:
                total = len(patches)  # type: ignore[arg-type]
            except TypeError:
                total = None
            progress = tqdm(total=total, desc="Aggregating thought patch", unit="patch")

        for patch in patches:
            delta = patch.delta
            a_t = patch.baseline
            deltas.append(delta)

            denom = torch.dot(a_t, a_t) + 1e-8
            rank_one = torch.outer(delta, a_t) / denom
            outer_sums = rank_one if outer_sums is None else outer_sums + rank_one
            count += 1
            if progress:
                progress.update(1)

        if progress:
            progress.close()

        if count == 0:
            raise ValueError("No patches provided to solver.")

        delta_bias = torch.stack(deltas, dim=0).mean(dim=0)
        scale = self.lambda_scale if self.lambda_scale is not None else 1.0 / count
        weight_delta = outer_sums * scale

        return ThoughtPatch(bias_delta=delta_bias, weight_delta=weight_delta)
