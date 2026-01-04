
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import os
import shutil
import sys

# Ensure the project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
try:
    from cartridges.transmutation.extractor import TokenPatchExtractor, TokenPatch
    from cartridges.transmutation.solver import ThoughtPatchSolver
    from cartridges.transmutation.pipeline import Transmuter
    from cartridges.transmutation.adapter import MultiLayerThoughtAdapter, register_thought_hook
except ImportError:
    print("Error: Could not import cartridges modules. converting to relative import...")
    # This might happen if run directly without installing the package
    pass

def test_transmutation_pipeline():
    print("=== Transmutation Pipeline Verification ===")
    print("Setting up mock environment...")
    hidden_size = 16
    layers_to_track = [0, 2] # Test arbitrary layers
    
    model = MockModel(hidden_size=hidden_size)
    
    # Create synthetic data
    elements = [
        MockDatasetElement(input_ids=torch.randint(0, 100, (10,))),
        MockDatasetElement(input_ids=torch.randint(0, 100, (15,))),
    ]
    dataset = MockDataset(elements)
    
    # 1. Extraction
    print("\n1. Testing Extraction (TokenPatchExtractor)...")
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
    print("\n2. Running Transmuter (Batch Extraction + Solver)...")
    solver = ThoughtPatchSolver()
    transmuter = Transmuter(extractor, solver)
    
    artifacts = transmuter.run(
        dataset=dataset,
        context_strip_fn=strip_fn,
        show_progress=True
    )
    
    print(f"Artifacts generated. Layers found: {list(artifacts.bias_deltas.keys())}")
    
    # Verifications
    if set(artifacts.bias_deltas.keys()) != set(layers_to_track):
        print(f"FAILED: Expected layers {layers_to_track}, got {list(artifacts.bias_deltas.keys())}")
        return
    print("SUCCESS: Artifacts contain correct layers.")

    # 3. Save
    print("\n3. Testing Save Artifacts...")
    os.makedirs("test_output", exist_ok=True)
    save_path = "test_output/transmuted_test.pt"
    artifacts.save(save_path)
    if os.path.exists(save_path):
        print("SUCCESS: File saved.")
    else:
        print("FAILED: File not found.")
        return
    
    # 4. Load Adapter
    print("\n4. Testing Load (MultiLayerThoughtAdapter)...")
    adapter = MultiLayerThoughtAdapter.from_pretrained(save_path)
    if set(adapter.adapters.keys()) == set(layers_to_track):
        print("SUCCESS: Adapter loaded with correct layers.")
    else:
        print(f"FAILED: Adapter has layers {list(adapter.adapters.keys())}")
        return
    
    # 5. Apply to Model
    print("\n5. Testing Application to Model...")
    
    def selector(m, i):
        # Map our tracked layer indices to modules in MockModel
        return m.layers[i]

    handles = adapter.apply(model, selector)
    print(f"Applied {len(handles)} hooks.")
    
    if len(handles) != len(layers_to_track):
        print(f"FAILED: Expected {len(layers_to_track)} hooks, got {len(handles)}")
        return

    # Verify hooks are actually working (run forward pass)
    inp = torch.randint(0, 100, (1, 5))
    try:
        _ = model(inp)
        print("SUCCESS: Forward pass with hooks successful.")
    except Exception as e:
        print(f"FAILED: Forward pass error: {e}")
    
    # Cleanup
    for h in handles:
        h.remove()
    shutil.rmtree("test_output")
    print("\n=== All Tests Passed ===")

if __name__ == "__main__":
    test_transmutation_pipeline()
