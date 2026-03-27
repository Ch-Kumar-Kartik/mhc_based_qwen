#!/usr/bin/env python3
"""Compare generation outputs between original and mHC models."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Compare generation between original and mHC models")
    parser.add_argument("--original", default="Qwen/Qwen3-0.6B", help="Original model path")
    parser.add_argument("--mhc", default="./qwen3-mhc", help="mHC model path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max new tokens to generate")
    args = parser.parse_args()

    # Test prompts
    prompts = [
        "The capital of France is",
        "def sort_list(lst):\n    ",
        "2 + 2 =",
        "The largest planet in our solar system is",
        "Hello, my name is",
        "To make a cup of coffee, first",
    ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.original)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading original model...")
    original = AutoModelForCausalLM.from_pretrained(
        args.original, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device
    )
    original.eval()

    print("Loading mHC model...")
    from src.qwen3_mhc_model import Qwen3MHCForCausalLM
    mhc = Qwen3MHCForCausalLM.from_pretrained(
        args.mhc,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    mhc.eval()

    print("\n" + "=" * 70)
    print("GENERATION COMPARISON (greedy decoding)")
    print("=" * 70)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            # Generate from original
            orig_out = original.generate(
                **inputs, 
                max_new_tokens=args.max_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)
            
            # Generate from mHC
            mhc_out = mhc.generate(
                **inputs, 
                max_new_tokens=args.max_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            mhc_text = tokenizer.decode(mhc_out[0], skip_special_tokens=True)
        
        match = "✓ SAME" if orig_text == mhc_text else "≈ DIFFERENT"
        
        print(f"\nPrompt: {repr(prompt)}")
        print(f"  Original: {orig_text}")
        print(f"  mHC:      {mhc_text}")
        print(f"  Status:   {match}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
