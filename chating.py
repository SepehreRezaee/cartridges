import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cartridges.transmutation.adapter import ThoughtAdapter, register_thought_hook

# 1) Load adapter produced by transmutation
ckpt = torch.load("outputs/transmuted_hf_adapter.pt", map_location="cpu")

adapter = ThoughtAdapter(
    bias_delta=ckpt["bias_delta"],
    weight_delta=ckpt["weight_delta"],
)

# 2) Load base model/tokenizer
model_name = ckpt.get("model_name", "Qwen/Qwen3-4b")
tokenizer_name = ckpt.get("tokenizer_name", model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()

# 3) Register the adapter as a forward hook (here on lm_head output)
handle = register_thought_hook(model, adapter, lambda m: m.lm_head)

# 4) Chat helper
def chat(prompt: str, max_new_tokens: int = 128, temperature: float = 0.2) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 5) Use it
print(chat("Summarize the key takeaways from the combined corpora.", max_new_tokens=150))

# 6) Remove hook when done
handle.remove()
