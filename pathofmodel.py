from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="tensorblock/Llama-3-OffsetBias-8B-GGUF",
    filename="Llama-3-OffsetBias-8B-Q5_K_M.gguf"
)

print(f"Local model path: {model_path}")
