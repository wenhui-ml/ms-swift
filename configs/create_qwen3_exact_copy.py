import os
import shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import save_file

def main():
    source_dir = "/home/ubuntu/llm_weights/Qwen3-0.6B"
    target_dir = "/home/ubuntu/wenhui/mag_gate/ms-swift/model_checkpoints/Qwen3-0.6B-standard"

    print(f"Creating exact copy of config from {source_dir} to {target_dir}")
    
    # 1. Create target directory and copy all files EXCEPT model.safetensors
    os.makedirs(target_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        if item.endswith(".safetensors") or item.endswith(".bin"):
            continue
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    print("Initializing random model weights based on exact config...")
    config = AutoConfig.from_pretrained(source_dir, trust_remote_code=True)
    # random initialization
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # ensure it's bfloat16 to match config
    model = model.to(torch.bfloat16)

    print("Extracting state dict...")
    state_dict = model.state_dict()
    
    # safetensors format requires tensors to be contiguous
    for k, v in list(state_dict.items()):
        state_dict[k] = v.clone().contiguous()
        
    print(f"Saving random model weights to {target_dir}/model.safetensors")
    save_file(state_dict, os.path.join(target_dir, "model.safetensors"), metadata={"format": "pt"})
    
    print("Done!")

if __name__ == '__main__':
    main()
