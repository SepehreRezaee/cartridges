from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


@dataclass
class ThoughtAdapter:
    """Applies a transmuted low-rank update at a chosen hook point.

    This is intentionally lightweight: it adds a residual projection
    using the aggregated weight/bias (Eq. 8, 24) from the transmuting
    prompts paper.
    """

    bias_delta: torch.Tensor
    weight_delta: torch.Tensor

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [*, hidden_size]
        return hidden + hidden @ self.weight_delta.T + self.bias_delta


class MultiLayerThoughtAdapter:
    """Container for multiple ThoughtAdapters targeting different layers."""

    def __init__(self, adapters: dict[int, ThoughtAdapter]):
        self.adapters = adapters

    @classmethod
    def from_pretrained(cls, path: str, map_location="cpu") -> MultiLayerThoughtAdapter:
        ckpt = torch.load(path, map_location=map_location)
        # Handle both old single-layer format and new multi-layer format
        if "bias_deltas" in ckpt:
            bias_deltas = ckpt["bias_deltas"]
            weight_deltas = ckpt["weight_deltas"]
            adapters = {}
            for layer_idx in bias_deltas:
                adapters[layer_idx] = ThoughtAdapter(
                    bias_delta=bias_deltas[layer_idx],
                    weight_delta=weight_deltas[layer_idx]
                )
            return cls(adapters)
        else:
            # Legacy fallback
             return cls({
                 -1: ThoughtAdapter(ckpt["bias_delta"], ckpt["weight_delta"])
             })
             
    def apply(
        self, 
        model: nn.Module, 
        layer_selector_fn: Callable[[nn.Module, int], nn.Module]
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Apply adapters to the model.
        
        Args:
            model: The root model.
            layer_selector_fn: Function that takes (model, layer_idx) and returns the submodule to hook.
                               Example: lambda m, i: m.model.layers[i].mlp
        """
        handles = []
        for layer_idx, adapter in self.adapters.items():
            # Handle negative indices if needed, though usually selector handles it
            try:
                target_module = layer_selector_fn(model, layer_idx)
                handle = register_thought_hook(target_module, adapter)
                handles.append(handle)
            except Exception as e:
                print(f"Failed to apply adapter to layer {layer_idx}: {e}")
        return handles


def register_thought_hook(
    module: nn.Module,
    adapter: ThoughtAdapter,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that applies the adapter to a target submodule."""

    def _hook(_mod, _inp, out):
        return adapter(out)

    return module.register_forward_hook(_hook)
