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
    ) -> Iterable[TokenPatch]:
        ...


@dataclass
class TransmutationArtifacts:
    """Container for the final transmuted weights and metadata."""

    bias_delta: torch.Tensor
    weight_delta: torch.Tensor
    metadata: dict


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
    ) -> TransmutationArtifacts:
        patches = list(
            self.extractor.extract(
                dataset,
                context_strip_fn=context_strip_fn,
                max_batches=max_batches,
            )
        )
        thought: ThoughtPatch = self.solver.solve(patches)
        metadata = extra_metadata or {}
        metadata.update(
            {
                "num_patches": len(patches),
                "lambda_scale": self.solver.lambda_scale,
            }
        )
        return TransmutationArtifacts(
            bias_delta=thought.bias_delta,
            weight_delta=thought.weight_delta,
            metadata=metadata,
        )
