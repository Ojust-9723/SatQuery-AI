"""
Inference script for Satellite Omni LISA model.
Supports both text tasks (captioning, VQA, classification) and
segmentation tasks (generates binary mask images).

Usage:
    # From HF Hub repo:
    python inference.py --model_repo rahuldshetty/satellite-omni-lisa --image path/to/satellite.jpg --prompt "Segment the buildings"
    # From local directory:
    python inference.py --model_dir ./trained_model --image path/to/satellite.jpg --prompt "Describe this satellite image"
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    SamModel,
    SamProcessor,
)
from peft import PeftModel


# ============================================================================
# Model Components (must match train.py)
# ============================================================================
class SegProjectionMLP(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
    
    def forward(self, x):
        return self.proj(x)


def load_model(model_dir: str = None, model_repo: str = None, device: str = "cuda"):
    """Load all components of the trained LISA model.
    
    Args:
        model_dir: Local directory path containing model files.
        model_repo: HF Hub repo ID (e.g., 'rahuldshetty/satellite-omni-lisa').
                    Uses snapshot_download to get clean local paths.
        device: Device to load model on.
    
    Exactly one of model_dir or model_repo must be provided.
    """
    if model_repo and not model_dir:
        from huggingface_hub import snapshot_download
        print(f"Downloading model from Hub: {model_repo}...")
        model_dir = snapshot_download(model_repo)
        print(f"Model files at: {model_dir}")
    
    if not model_dir:
        raise ValueError("Either model_dir or model_repo must be provided")
    
    # Load config
    with open(os.path.join(model_dir, "model_config.json")) as f:
        config = json.load(f)
    
    vlm_model_id = config["vlm_model_id"]
    sam_model_id = config["sam_model_id"]
    vlm_hidden_size = config["vlm_hidden_size"]
    sam_prompt_dim = config["sam_prompt_dim"]
    seg_token = config["seg_token"]
    
    # Load processor — try saved tokenizer first, fallback to base + re-add token
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        try:
            processor = AutoProcessor.from_pretrained(tokenizer_dir, trust_remote_code=True)
            seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
        except Exception:
            processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
            processor.tokenizer.add_tokens([seg_token], special_tokens=True)
            seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)
    else:
        processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
        processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)
    
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load base VLM + LoRA adapter
    dtype = torch.float16 if device != "cpu" else torch.float32
    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
        vlm_model_id, torch_dtype=dtype, device_map=device,
    )
    vlm.resize_token_embeddings(len(processor.tokenizer))
    print("SEG token ID:", seg_token_id)
    print("Model vocab size:", vlm.get_input_embeddings().weight.shape[0])
    vlm = PeftModel.from_pretrained(vlm, os.path.join(model_dir, "vlm_lora"))
    vlm.eval()
    
    # Load SAM
    sam = SamModel.from_pretrained(sam_model_id, torch_dtype=torch.float32).to(device)
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    
    # Load fine-tuned SAM mask decoder weights
    mask_decoder_state = torch.load(
        os.path.join(model_dir, "sam_mask_decoder.pt"), map_location=device
    )
    sam.mask_decoder.load_state_dict(mask_decoder_state)
    sam.eval()
    
    # Load seg projector
    seg_projector = SegProjectionMLP(in_dim=vlm_hidden_size, out_dim=sam_prompt_dim).to(device)
    projector_state = torch.load(
        os.path.join(model_dir, "seg_projector.pt"), map_location=device
    )
    seg_projector.load_state_dict(projector_state)
    seg_projector.eval()
    
    return {
        "vlm": vlm,
        "sam": sam,
        "sam_processor": sam_processor,
        "seg_projector": seg_projector,
        "processor": processor,
        "seg_token_id": seg_token_id,
        "seg_token": seg_token,
        "device": device,
    }


@torch.no_grad()
def generate_text(model_dict, image: Image.Image, prompt: str, max_new_tokens: int = 512):
    """Generate text response (captioning, VQA, classification)."""
    processor = model_dict["processor"]
    vlm = model_dict["vlm"]
    device = model_dict["device"]
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    
    output_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    
    return response


@torch.no_grad()
def generate_segmentation(model_dict, image: Image.Image, prompt: str):
    """Generate text + segmentation mask."""

    processor = model_dict["processor"]
    vlm = model_dict["vlm"]
    sam = model_dict["sam"]
    sam_processor = model_dict["sam_processor"]
    seg_projector = model_dict["seg_projector"]
    seg_token_id = model_dict["seg_token_id"]
    device = model_dict["device"]

    # Build the same assistant response format used during training.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I identified several building instances in this image. <SEG>",
                }
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    # Run the completed assistant response through the VLM.
    vlm_outputs = vlm(
        **inputs,
        output_hidden_states=True,
    )

    # Decode the assistant response.
    text_response = processor.tokenizer.decode(
        inputs["input_ids"][0],
        skip_special_tokens=True,
    )

    # Find the <SEG> token.
    input_ids = inputs["input_ids"][0]
    seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)[0]

    mask_image = None

    if len(seg_positions) > 0:
        # Get hidden state corresponding to <SEG>.
        seg_position = seg_positions[0].item()
        h_seg = vlm_outputs.hidden_states[-1][:, seg_position, :]

        # Project VLM embedding to SAM prompt.
        sparse_prompt = seg_projector(h_seg.float())
        sparse_prompt = sparse_prompt.unsqueeze(1).unsqueeze(1)

        # Encode image with SAM.
        sam_inputs = sam_processor(
            images=[image],
            return_tensors="pt",
        ).to(device)

        sam_image_embeddings = sam.get_image_embeddings(
            sam_inputs["pixel_values"]
        )

        # Get positional and dense embeddings.
        image_pos = sam.get_image_wide_positional_embeddings()

        _, dense_embeddings = sam.prompt_encoder(
            input_points=None,
            input_labels=None,
            input_boxes=None,
            input_masks=None,
        )

        # Decode segmentation mask.
        mask_preds, iou_preds = sam.mask_decoder(
            image_embeddings=sam_image_embeddings,
            image_positional_embeddings=image_pos,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Convert to binary mask image.
        mask = torch.sigmoid(mask_preds.squeeze()).cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        mask_image = Image.fromarray(mask).resize(
            image.size,
            Image.NEAREST,
        )

    return text_response, mask_image


def main():
    parser = argparse.ArgumentParser(description="Satellite Omni LISA Inference")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to local trained model directory")
    parser.add_argument("--model_repo", type=str, default=None, help="HF Hub repo ID (e.g., rahuldshetty/satellite-omni-lisa)")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--task", type=str, choices=["text", "segment", "auto"], default="auto",
                       help="Task type: text (captioning/VQA), segment (segmentation), auto (detect from prompt)")
    parser.add_argument("--output_mask", type=str, default="mask_output.png", help="Path to save mask output")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    if not args.model_dir and not args.model_repo:
        parser.error("Either --model_dir or --model_repo must be provided")
    
    # Load model
    src = args.model_dir or args.model_repo
    print(f"Loading model from {src}...")
    model_dict = load_model(model_dir=args.model_dir, model_repo=args.model_repo, device=args.device)
    
    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.size}")
    
    # Determine task
    seg_keywords = ["segment", "segmentation", "mask", "detect and segment", "instance"]
    is_seg = args.task == "segment" or (
        args.task == "auto" and any(kw in args.prompt.lower() for kw in seg_keywords)
    )
    
    if is_seg:
        print("Task: Segmentation")
        text_response, mask_image = generate_segmentation(model_dict, image, args.prompt)
        print(f"Response: {text_response}")
        if mask_image is not None:
            mask_image.save(args.output_mask)
            print(f"Mask saved to: {args.output_mask}")
        else:
            print("No segmentation mask generated (model did not produce <SEG> token)")
    else:
        print("Task: Text generation")
        response = generate_text(model_dict, image, args.prompt)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
