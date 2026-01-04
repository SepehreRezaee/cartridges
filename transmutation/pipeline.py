from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import torch

from cartridges.datasets import TrainDataset, DatasetElement
from transmutation.extractor import TokenPatchExtractor, TokenPatch
from transmutation.solver import ThoughtPatch, ThoughtPatchSolver


class PatchExtractor(Protocol):
    """Protocol for extractors to encourage dependency inversion (SOLID: DIP)."""

    def extract(
        self,
        dataset: TrainDataset,
        context_strip_fn: Optional[callable] = None,
        max_batches: Optional[int] = None,
        show_progress: bool = False,
    ) -> Iterable[TokenPatch]:
        ...


@dataclass
class TransmutationArtifacts:
    """Container for the final transmuted weights and metadata."""

    bias_deltas: dict[int, torch.Tensor]
    weight_deltas: dict[int, torch.Tensor]
    metadata: dict

    def save(self, path: str):
        """Save artifacts to a file."""
        torch.save({
            "bias_deltas": self.bias_deltas,
            "weight_deltas": self.weight_deltas,
            "metadata": self.metadata
        }, path)

class Transmuter:
    """Coordinates extraction and solving (SOLID: SRP, DIP)."""

    def __init__(
        self,
        extractor: PatchExtractor,
        solver: ThoughtPatchSolver,
    ) -> None:
        self.extractor = extractor
        self.solver = solver

    def run(
        self,
        dataset: TrainDataset,
        context_strip_fn: Optional[callable] = None,
        max_batches: Optional[int] = None,
        extra_metadata: Optional[dict] = None,
        show_progress: bool = False,
    ) -> TransmutationArtifacts:
        patches = list(
            self.extractor.extract(
                dataset,
                context_strip_fn=context_strip_fn,
                max_batches=max_batches,
                show_progress=show_progress,
            )
        )
        
        # solver.solve now returns dict[int, ThoughtPatch]
        thought_patches: dict[int, ThoughtPatch] = self.solver.solve(patches, show_progress=show_progress)
        
        bias_deltas = {}
        weight_deltas = {}
        
        for layer_idx, patch in thought_patches.items():
            bias_deltas[layer_idx] = patch.bias_delta
            weight_deltas[layer_idx] = patch.weight_delta
            
        metadata = extra_metadata or {}
        metadata.update(
            {
                "num_patches": len(patches),
                "lambda_scale": self.solver.lambda_scale,
                "layers": list(thought_patches.keys()),
            }
        )
        return TransmutationArtifacts(
            bias_deltas=bias_deltas,
            weight_deltas=weight_deltas,
            metadata=metadata,
        )
