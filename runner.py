# transmute_hf.py (run from repo root: cartridges/)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cartridges.datasets import DataSource, TrainDataset
from transmutation.extractor import TokenPatchExtractor
from transmutation.solver import ThoughtPatchSolver
from transmutation.pipeline import Transmuter
from transmutation.adapter import ThoughtAdapter, register_thought_hook

# HF datasets
sources = [
    DataSource(path="hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0", type="hf"),
    DataSource(path="hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-0", type="hf"),
]

model_name = "Qwen/Qwen3-4b"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

dataset = TrainDataset(
    TrainDataset.Config(
        data_sources=sources,
        packed_seq_length=2048,
        targets="tokens",
    ),
    tokenizer=tokenizer,
    seed=0,
)

extractor = TokenPatchExtractor(model=model, tokenizer=tokenizer, device=device)
solver = ThoughtPatchSolver()
transmuter = Transmuter(extractor=extractor, solver=solver)

artifacts = transmuter.run(dataset)
torch.save(
    {"bias_delta": artifacts.bias_delta, "weight_delta": artifacts.weight_delta, "metadata": artifacts.metadata},
    "outputs/transmuted_hf_adapter.pt",
)
print("Saved adapter to outputs/transmuted_hf_adapter.pt")

# Use the adapter for chat (no cartridges needed)
adapter = ThoughtAdapter(artifacts.bias_delta, artifacts.weight_delta)
handle = register_thought_hook(model, adapter, lambda m: m.lm_head)

prompt = "Summarize the most critical facts from the combined corpora."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, temperature=0.2)
print(tokenizer.decode(out[0], skip_special_tokens=True))

handle.remove()
