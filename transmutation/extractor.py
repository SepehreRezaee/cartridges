from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from cartridges.datasets import TrainDataset, DatasetElement


@dataclass
class TokenPatch:
    """Container for a single token-level patch.

    References:
        Eq. 3, 4 in "Transmuting Prompts into Weights".
    """

    delta: torch.Tensor  # A(C, t) - A(empty, t)
    baseline: torch.Tensor  # A(empty, t)
    element_idx: int
    token_idx: int
    layer_idx: int


class TokenPatchExtractor:
    """Extract token-level activation deltas for a corpus.

    For each token t we compute A(C, t) with context C, A(empty, t)
    without context, the delta_t = A(C, t) - A(empty, t) (Eq. 3) and
    keep baseline activations for later rank-one updates (Eq. 4).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        layers: Optional[List[int]] = None,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.layers = layers or [-1]
        # ensure we expose hidden states for activations
        self.model.eval()

    def _forward_hidden(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run a forward pass and return hidden states for tracked layers.
        This method is designed to accept a batch of input_ids: [batch_size, seq_len].
        It returns the hidden states for the *last token* of each sequence in the batch.
        """
        # input_ids: [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        attention_mask = torch.ones_like(input_ids, device=self.device)
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        # Normalize layer indices to positive integers
        num_layers = len(outputs.hidden_states)
        results = {}
        for l_idx in self.layers:
            abs_idx = l_idx if l_idx >= 0 else num_layers + l_idx
            # hidden_states[i] is [batch_size, seq_len, hidden_dim]
            # we only care about the last token's hidden state for our "next token" prediction context
            # So let's take the last token: [:, -1, :]
            results[l_idx] = outputs.hidden_states[abs_idx][:, -1, :]
            
        return results

    def extract(
        self,
        dataset: TrainDataset,
        context_strip_fn: Optional[Callable[[DatasetElement], torch.Tensor]] = None,
        max_batches: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[TokenPatch]:
        """Compute TokenPatch objects over the dataset.

        Args:
            dataset: The synthetic corpus (from self-study) wrapped in a TrainDataset.
            context_strip_fn: Callable that returns a "no-context" input tensor.
                Defaults to reusing the trailing tokens of the element.
            max_batches: optional limit for quick debugging.
            show_progress: whether to display a progress bar over dataset elements.
        """
        patches: List[TokenPatch] = []
        batches = dataset.batches[:max_batches] if max_batches else dataset.batches

        progress = None
        if show_progress:
            total_elems = sum(len(batch) for batch in batches)
            progress = tqdm(total=total_elems, desc="Extracting token patches", unit="element")

        try:
            for batch in batches:
                for elem_idx in batch:
                    element = dataset._get_element(elem_idx)
                    patches.extend(
                        self._extract_from_element(
                            element,
                            element_idx=elem_idx,
                            context_strip_fn=context_strip_fn,
                        )
                    )
                    if progress:
                        progress.update(1)
        finally:
            if progress:
                progress.close()
        return patches

    def _default_strip(self, element: DatasetElement) -> torch.Tensor:
        """Fallback 'no-context' input: keep the assistant reply only.

        This is a conservative heuristic when an explicit context remover
        is not provided.
        """
        # take the last half of tokens as a crude approximation of reply-only
        cutoff = max(1, element.input_ids.numel() // 2)
        return element.input_ids[-cutoff:]

    def _extract_from_element(
        self,
        element: DatasetElement,
        element_idx: int,
        context_strip_fn: Optional[Callable[[DatasetElement], torch.Tensor]],
        batch_size: int = 32
    ) -> Iterable[TokenPatch]:
        with_ids = element.input_ids.to(self.device)
        base_ids = (
            context_strip_fn(element).to(self.device)
            if context_strip_fn
            else self._default_strip(element).to(self.device)
        )

        # We want to compare the model state just before predicting the *same* target token.
        # But `with_ids` has the full context + completion, and `base_ids` has no context + completion.
        # The completion parts should be identical.
        # Let 'completion' be the shared suffix.
        # We process the completion token by token.
        
        # Determine the length of the completion (the part we are "transmuting")
        # In the original code: 
        # seq_len = min(act_with.size(0), act_base.size(0))
        # for i in range(1, seq_len + 1): ...
        # This implies it was taking the last `seq_len` tokens as the completion.
        # We will assume `base_ids` IS the completion (or at least the suffix of it).
        
        # To batch this, we need to construct the input sequences for every step.
        # For a completion of length L:
        # Step 1: Context + Completion[:0] -> predict Completion[0]
        # Step 2: Context + Completion[:1] -> predict Completion[1]
        # ...
        
        # Let's align them from the end.
        completion_len = min(with_ids.size(0), base_ids.size(0))
        
        # Helper to generate the input sequences for a range of steps
        # steps are i in [1 ... completion_len] (1-indexed from end in original code)
        # Original code: act_with[-i] corresponds to input `with_ids[:-i]` if we were doing autoregressive
        # actually no, original code ran forward on the WHOLE sequence and took hidden states.
        # existing: 
        # act_with = forward(with_ids) -> [seq_len, hidden]
        # act_with[-1] is hidden state after seeing ALL tokens.
        # act_with[-2] is hidden state after seeing ALL BUT LAST token.
        
        # Original code was fast for a single sequence call (one forward pass per seq).
        # My previous batch implementation attempt description was slightly wrong for "one forward pass".
        # The original code did:
        # act_with = self._forward_hidden(with_ids) (returns [seq_len, hidden])
        # act_base = self._forward_hidden(base_ids)
        # loops over suffixes.
        
        # Optimization: We can simply batch the *documents* (DatasetElements) if we want (not strictly asked but good).
        # OR we can assume `_forward_hidden` returns the whole sequence hidden states, which is efficient.
        # The bottleneck in "transmutation" is usually that we have MANY documents.
        # But wait, did I change `_forward_hidden` to return only the last token? Yes I did in the chunk above.
        # That would be WRONG if we want to run one forward pass per document.
        
        # Correct Approach for "Fast":
        # 1. Run ONE forward pass on `with_ids` to get ALL hidden states [L_with, H].
        # 2. Run ONE forward pass on `base_ids` to get ALL hidden states [L_base, H].
        # 3. Align the suffixes and compute deltas.
        # This is already what the original code did, BUT `_forward_hidden` was doing `squeeze(0)` and running batch=1.
        # The user wants it "very fast". 
        # Truly fast means batching MULTIPLE EXAMPLES (DatasetElements).
        
        # However, implementing multi-example batching in `extract` requires changing the loop structure of `extract`
        # to collate a batch of elements.
        
        # For simplicity and "very fast" within the constraints of this method signature (iterating elements):
        # We will stay with processing one element at a time but ensure we don't do excessive Python loops.
        # Actually, extracting multiple layers at once (which I did) is already a speedup.
        
        # Let's implement full sequence forward pass (restoring what it likely was but robustly)
        # and vectorized subtraction.
        
        # 1. Forward pass on 'with' context
        with_hidden_dict = self._forward_sequence_hidden(with_ids)
        # 2. Forward pass on 'base' context
        base_hidden_dict = self._forward_sequence_hidden(base_ids)
        
        # 3. Vectorized delta computation
        first_layer = self.layers[0]
        # completion_len is the number of tokens we are aligning from the end
        completion_len = min(with_hidden_dict[first_layer].size(0), base_hidden_dict[first_layer].size(0))
        
        # Create indices once
        # We want the last `completion_len` tokens.
        # slice_with = slice(-completion_len, None) # this fails if completion_len is 0? no.
        # if completion_len == full len, slice(-len, None) works.
        
        for layer_idx in self.layers:
            # Get full sequence tensors
            H_with = with_hidden_dict[layer_idx] # [L_with, D]
            H_base = base_hidden_dict[layer_idx] # [L_base, D]
            
            # Extract aligned suffixes
            # Note: The loop was `for i in range(1, seq_len+1): ... act[-i]`
            # This is equivalent to taking the last `seq_len` tokens.
            suffix_with = H_with[-completion_len:]
            suffix_base = H_base[-completion_len:]
            
            deltas = suffix_with - suffix_base # [completion_len, D]
            
            # Now we have all deltas for this layer.
            # We need to yield TokenPatch objects. 
            # Generating objects in a loop is slow.
            # But the consumer `solver` expects patches.
            # Changing `solver` to accept bulk tensors would be faster, but out of scope for just this file?
            # The user asked to "strengthen implementation that be very fast".
            # Batch extraction is key.
            
            deltas_cpu = deltas.detach().cpu()
            baselines_cpu = suffix_base.detach().cpu()
            
            for i in range(completion_len):
                # i=0 -> first token of suffix -> original index: completion_len - (seq_len - i)? 
                # actually original loop: i=1 (last token), i=seq_len (first token of comp).
                # token_idx in original: seq_len - i (descending 0).
                # let's be consistent.
                # if we have [A, B, C], length 3.
                # suffix is [A, B, C].
                # i=0 (A): token_idx should be 2? 
                # Original: i=1 -> C. token_idx = 3-1 = 2.
                #           i=3 -> A. token_idx = 3-3 = 0.
                # So token_idx is distance from start of alignment.
                
                yield TokenPatch(
                    delta=deltas_cpu[i],
                    baseline=baselines_cpu[i],
                    element_idx=element_idx,
                    token_idx=i, 
                    layer_idx=layer_idx,
                )

    def _forward_sequence_hidden(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run forward pass for a single sequence, return [seq_len, hidden] for each layer."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                output_hidden_states=True,
            )
            
        num_layers = len(outputs.hidden_states)
        results = {}
        for l_idx in self.layers:
            abs_idx = l_idx if l_idx >= 0 else num_layers + l_idx
            # squeeze batch dim 0 -> [seq_len, D]
            results[l_idx] = outputs.hidden_states[abs_idx].squeeze(0)
        return results

