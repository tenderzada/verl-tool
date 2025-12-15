#!/usr/bin/env python3
"""
Quick Inference Script for Pixel Reasoner
This script allows you to test the Pixel Reasoner model on single images/videos interactively.

Usage:
    python examples/train/pixel_reasoner/quick_inference.py \
        --model_path VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
        --image_path /path/to/your/image.jpg \
        --question "What is shown in this image?"
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Quick inference with Pixel Reasoner")
    parser.add_argument(
        "--model_path",
        type=str,
        default="VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100",
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    return parser.parse_args()

def load_model_and_processor(model_path, device):
    """Load the model and processor"""
    print(f"Loading model from {model_path}...")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print("Model loaded successfully!")
    return model, processor

def create_prompt(question):
    """Create the prompt with tool instructions"""
    system_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}}, "required": ["target_frames"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    guideline = """Guidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `select_frames`. For an image, we can look closer by `crop_image`. Reason with the visual information step by step, and put your final answer within \\boxed{}."""

    return f"{system_prompt}\n\n{guideline}\n\nUser: {question}\nAssistant:"

def inference(model, processor, image_path, question, args):
    """Run inference on a single image"""
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    print(f"Image loaded: {image.size}")

    # Create prompt
    prompt = create_prompt(question)

    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Generate response
    print("\nGenerating response...")
    print("=" * 80)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True if args.temperature > 0 else False,
        )

    # Decode response
    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return response

def main():
    args = parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    # Load model
    model, processor = load_model_and_processor(args.model_path, args.device)

    # Run inference
    try:
        response = inference(model, processor, args.image_path, args.question, args)

        print("\n" + "=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        # Try to extract answer from \boxed{}
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            print("\nFINAL ANSWER:")
            print(boxed_match.group(1))

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
