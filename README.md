# Cartridges: Long Context Without Long Context

<div align="center">
    <img src="assets/banner.png" height="100" alt="Cartridges logo"/>
</div>

Lightweight, corpus-specific KV caches trained via self-study, plus a “transmutation” path that bakes cartridges into model weights (bias/low-rank deltas).

## What’s Inside
- **Self-study synthesis** (`examples/arxiv/arxiv_synthesize.py`): generate synthetic conversations about a corpus.
- **Cartridge training** (`examples/arxiv/arxiv_train.py`): distill those conversations into a small KV cache.
- **Serving cartridges**: plug KV caches into [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) via `cartridges.clients.tokasaurus`.
- **Transmutation (new)**: convert synthetic data into weight/bias deltas (ThoughtAdapter) so you can run without KV caches.
- **Benchmarks**: direct cartridges vs transmuted adapters (`examples/benchmarks/transmute_vs_cartridge.py`).

## Quickstart
### Install
```bash
git clone https://github.com/HazyResearch/cartridges
cd cartridges
python -m pip install uv
uv pip install -e .
```
Set required env vars (e.g. in your shell profile):
```bash
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export HF_TOKEN=<your-hf-token>        # if accessing private HF
export WANDB_API_KEY=<your-wandb-key>  # if logging to wandb
```

### Synthesize + Train a Cartridge (example)
```bash
python examples/arxiv/arxiv_synthesize.py
python examples/arxiv/arxiv_train.py    # produces a KV cache artifact
```

### Serve a Cartridge (Tokasaurus)
Start Tokasaurus (geoff/cartridges branch) with your model and KV capacity, then query:
```python
from cartridges.clients.tokasaurus import TokasaurusClient
client = TokasaurusClient.Config(
    model_name="Qwen/Qwen3-4b",
    url="http://localhost:10210",
    cartridges=[{"id": "hazyresearch/cartridge-wauoq23f", "source": "huggingface"}],
).instantiate()
```

## Transmutation: Prompts → Weights
Transmutation converts self-study synthetic data into a low-rank “ThoughtAdapter” (bias + weight deltas). Useful when you want cartridge knowledge without runtime KV caches.

### Transmute one corpus
```bash
python -m examples.transmute_corpus \
  --data-path /path/to/synthetic.parquet \
  --model-name Qwen/Qwen3-4b \
  --output-path outputs/transmuted_adapter.pt \
  --device cuda
```

### Transmute multiple cartridges together
```bash
python -m examples.transmute_cartridges \
  --data-paths /path/to/cart1.parquet /path/to/cart2.parquet \
  --model-name Qwen/Qwen3-4b \
  --output-path outputs/transmuted_combo.pt \
  --device cuda
```

### Use the transmuted adapter for chat (no cartridges needed)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cartridges.transmutation.adapter import ThoughtAdapter, register_thought_hook

ckpt = torch.load("outputs/transmuted_adapter.pt", map_location="cpu")
adapter = ThoughtAdapter(ckpt["bias_delta"], ckpt["weight_delta"])

model_name = ckpt.get("model_name", "Qwen/Qwen3-4b")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()

handle = register_thought_hook(model, adapter, lambda m: m.lm_head)
inputs = tokenizer("Summarize the corpus.", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, temperature=0.2)
print(tokenizer.decode(out[0], skip_special_tokens=True))
handle.remove()
```

## Benchmarks (cartridge vs transmuted)
`examples/benchmarks/transmute_vs_cartridge.py` runs:
- LongHealth recall (cartridge vs adapter)
- MTOB skill (adapter)
- Throughput tokens/sec (both)

Example:
```bash
python -m examples.benchmarks.transmute_vs_cartridge \
  --model-name Qwen/Qwen3-4b \
  --adapter-path outputs/transmuted_adapter.pt \
  --cartridge-ids hazyresearch/cartridge-wauoq23f \
  --tokasaurus-url http://localhost:10210
```

## Key Files & Modules
- `cartridges/datasets.py` — wraps synthetic conversations into train/eval datasets.
- `cartridges/train.py` — cartridge training loop.
- `cartridges/clients/tokasaurus.py` — Tokasaurus client with cartridge support.
- `cartridges/transmutation/` — prompt-to-weights pipeline:
  - `extractor.py`: Token-level deltas (Eq. 3/4).
  - `solver.py`: Aggregate into bias/weight deltas (Eq. 8/24).
  - `adapter.py`: Apply deltas as a hook.
  - `pipeline.py`: Orchestration helpers.

## Environment Variables
- `CARTRIDGES_DIR` — path to this repo.
- `CARTRIDGES_OUTPUT_DIR` — where outputs/artifacts are written.
- `HF_TOKEN` — for private HF datasets/models (optional).
- `WANDB_API_KEY` — for logging (optional).

## Citation
If you use this code, please cite:
```
@article{eyuboglu2025cartridges,
  title={Cartridges: Lightweight and general-purpose long context representations via self-study},
  author={Eyuboglu, Sabri and Ehrlich, Ryan and Arora, Simran and Guha, Neel and Zinsley, Dylan and Liu, Emily and Tennien, Will and Rudra, Atri and Zou, James and Mirhoseini, Azalia and others},
  journal={arXiv preprint arXiv:2506.06266},
  year={2025}
}
```
