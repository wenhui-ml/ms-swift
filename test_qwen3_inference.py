#!/usr/bin/env python
"""Test inference for Qwen3-0.6B-standard baseline model.

Usage:
    python test_qwen3_inference.py
    python test_qwen3_inference.py --model_path model_checkpoints/Qwen3-0.6B-standard
    python test_qwen3_inference.py --device cuda
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-0.6B-standard model inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_checkpoints/Qwen3-0.6B-standard",
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # Resolve model path
    model_path = args.model_path
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    # ==================== Load Config ====================
    print(f"{'='*60}")
    print(f"Loading config from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"  model_type:           {config.model_type}")
    print(f"  hidden_size:          {config.hidden_size}")
    print(f"  num_hidden_layers:    {config.num_hidden_layers}")
    print(f"  num_attention_heads:  {config.num_attention_heads}")
    print(f"  num_key_value_heads:  {config.num_key_value_heads}")
    print(f"  head_dim:             {config.head_dim}")
    print(f"  intermediate_size:    {config.intermediate_size}")
    print(f"  vocab_size:           {config.vocab_size}")
    print(f"{'='*60}")

    # ==================== Load Tokenizer ====================
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  vocab_size: {tokenizer.vocab_size}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  pad_token set to eos_token: {tokenizer.eos_token}")

    # ==================== Load Model ====================
    print(f"\nLoading model to {args.device} ({args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully!")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # ==================== Test Prompts ====================
    test_prompts = [
        "The capital of France is",
        "In a distant galaxy, there was",
        "def fibonacci(n):\n",
        "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    ]

    print(f"\n{'='*60}")
    print(f"Generating with max_new_tokens={args.max_new_tokens}")
    print(f"{'='*60}")

    model.eval()
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {repr(prompt)}")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_only = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Output: {repr(generated_text)}")
        print(f"New tokens ({len(new_tokens)}): {repr(generated_only)}")

    print(f"\n✅ All tests passed!")


if __name__ == "__main__":
    main()
