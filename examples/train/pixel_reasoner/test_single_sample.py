#!/usr/bin/env python3
"""
Simple test script to verify model loading and single-sample inference
Bypasses verl framework complexity
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*50)
print("Step 1/5: Importing libraries...")
print("="*50)

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import sys

model_path = "/mnt/data/pixel_reasoner_3b"

print("\n" + "="*50)
print("Step 2/5: Loading tokenizer...")
print("="*50)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Step 3/5: Loading model...")
print("="*50)
print("This may take 1-2 minutes...")

try:
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Model loaded successfully")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Step 4/5: Checking GPU memory...")
print("="*50)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        free = total - cached
        print(f"  GPU {i}:")
        print(f"    Total: {total:.2f} GB")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Cached: {cached:.2f} GB")
        print(f"    Free: {free:.2f} GB")
else:
    print("  No CUDA devices available")

print("\n" + "="*50)
print("Step 5/5: Testing single inference...")
print("="*50)

try:
    test_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([test_input], return_tensors="pt").to(model.device)

    print("  Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  ✓ Response generated: {response[:100]}...")

except Exception as e:
    print(f"  ✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("SUCCESS: Model is working correctly!")
print("="*50)
print("\nYou can now proceed with full dataset inference.")
