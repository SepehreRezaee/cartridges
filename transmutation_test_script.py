
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import os
import shutil

# Mock classes to avoid full dependencies and download large models
@dataclass
class MockDatasetElement:
    input_ids: torch.Tensor
    metadata: list = None

class MockDataset:
    def __init__(self, elements):
        self.elements = elements
        self.batches = [[i] for i in range(len(elements))]
    
    def _get_element(self, idx):
        return self.elements[idx]

class MockModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        # Create dummy hidden states: [batch, seq_len, hidden]
        batch_size, seq_len = input_ids.shape
        hidden_states = []
        
        # Base state dependent on input for reproducibility
        # Just use embedding-like lookup for simplicity
        h = torch.randn(batch_size, seq_len, self.hidden_size) + input_ids.unsqueeze(-1).float()
        
        hidden_states.append(h)
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
            
        @dataclass
        class Output:
            hidden_states: tuple
            
        return Output(hidden_states=tuple(hidden_states))

# Import the actual classes to test
from cartridges.transmutation.extractor import TokenPatchExtractor, TokenPatch
from cartridges.transmutation.solver import ThoughtPatchSolver
from cartridges.transmutation.pipeline import Transmuter
from cartridges.transmutation.adapter import MultiLayerThoughtAdapter, register_thought_hook

def test_transmutation_pipeline():
    print("Setting up mock environment...")
    hidden_size = 16
    layers_to_track = [0, 2] # Test arbitrary layers
    
    model = MockModel(hidden_size=hidden_size)
    
    # Create synthetic data
    # element 1: 10 tokens
    # element 2: 15 tokens
    elements = [
        MockDatasetElement(input_ids=torch.randint(0, 100, (10,))),
        MockDatasetElement(input_ids=torch.randint(0, 100, (15,))),
    ]
    dataset = MockDataset(elements)
    
    # 1. Extraction
    print("1. Testing Extraction...")
    extractor = TokenPatchExtractor(
        model=model,
        tokenizer=None, # Not needed for this test
        layers=layers_to_track,
        device="cpu"
    )
    
    # Mock context strip function (just remove first token)
    def strip_fn(elem):
        return elem.input_ids[1:]
    
    # 2. Pipeline Run (Extract + Solve)
    print("2. Running Transmuter...")
    solver = ThoughtPatchSolver()
    transmuter = Transmuter(extractor, solver)
    
    artifacts = transmuter.run(
        dataset=dataset,
        context_strip_fn=strip_fn,
        show_progress=True
    )
    
    print(f"Artifacts generated. Layers found: {list(artifacts.bias_deltas.keys())}")
    assert set(artifacts.bias_deltas.keys()) == set(layers_to_track)
    assert set(artifacts.weight_deltas.keys()) == set(layers_to_track)
    
    # 3. Save
    print("3. Testing Save...")
    os.makedirs("test_output", exist_ok=True)
    save_path = "test_output/transmuted_test.pt"
    artifacts.save(save_path)
    assert os.path.exists(save_path)
    
    # 4. Load Adapter
    print("4. Testing Load Adapter...")
    adapter = MultiLayerThoughtAdapter.from_pretrained(save_path)
    assert set(adapter.adapters.keys()) == set(layers_to_track)
    
    # 5. Apply to Model
    print("5. Testing Apply to Model...")
    
    # Selector: target the linear layer in the mock model
    # MockModel structure: model.layers[i]
    # Note: extractor index 0 maps to hidden_states[0] which is usually embedding output/input to layer 0.
    # But usually we hook the output of a layer. 
    # For this test, let's just hook the Linear modules we defined.
    
    def selector(m, i):
        # Map our tracked layer indices to modules
        # Our MockModel has random linear layers.
        # Let's just say we hook the i-th layer
        return m.layers[i]

    handles = adapter.apply(model, selector)
    print(f"Applied {len(handles)} hooks.")
    assert len(handles) == len(layers_to_track)
    
    # Verify hooks are actually working (run forward pass)
    inp = torch.randint(0, 100, (1, 5))
    _ = model(inp)
    print("Forward pass with hooks successful.")
    
    # Cleanup
    for h in handles:
        h.remove()
    shutil.rmtree("test_output")
    print("Test Complete: SUCCESS")

if __name__ == "__main__":
    test_transmutation_pipeline()
