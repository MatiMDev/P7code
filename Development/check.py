from safetensors.torch import load_file

file_path = "./LoRA-SAM/lora_weights/lora_rank2.safetensors"

try:
    lora_weights = load_file(file_path)
    print("Keys in the LoRA weights file:", lora_weights.keys())
    for key, tensor in lora_weights.items():
        print(f"{key}: shape={tensor.shape}, min={tensor.min()}, max={tensor.max()}")
except Exception as e:
    print(f"Error loading file: {e}")
