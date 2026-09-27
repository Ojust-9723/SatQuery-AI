"""
Dataset Evaluation Script for Satellite Omni LISA
==================================================
Samples from each task type in the dataset, runs inference through the trained model,
and generates a REPORT.md with side-by-side comparisons of expected vs actual outputs.

Usage:
    python infer_dataset.py --model_repo rahuldshetty/satellite-omni-lisa
    python infer_dataset.py --model_repo rahuldshetty/satellite-omni-lisa --samples_per_task 5 --output_dir ./eval_results
    python infer_dataset.py --model_repo rahuldshetty/satellite-omni-lisa --split test --device cpu

Output:
    <output_dir>/
    ├── REPORT.md                  # Markdown report with all results
    ├── images/                    # Saved input images
    │   ├── sample_001_input.png
    │   ├── sample_001_pred_mask.png
    │   ├── sample_001_gt_mask.png
    │   └── ...
    └── results.json               # Machine-readable results
"""

import os
import re
import json
import random
import logging
import argparse
import tempfile
import shutil
from datetime import datetime
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from datasets import load_dataset

from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    SamModel,
    SamProcessor,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Constants (must match train.py)
# ============================================================================
DATASET_ID = "rahuldshetty/satellite-multitask-omni"
SEG_TOKEN = "<SEG>"
SEGMENTATION_TASKS = {"instance_segmentation"}

# ============================================================================
# Model Components (mirrors train.py)
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


# ============================================================================
# Polygon / Mask Utilities (from train.py)
# ============================================================================
def parse_polygons_from_text(text: str, img_w: int, img_h: int):
    polygons = []
    pattern = r'polygon\s*\[(.*?)\]'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        coords = re.findall(r'\((\d+),\s*(\d+)\)', match)
        if coords:
            poly = [(int(x) / 1000.0 * img_w, int(y) / 1000.0 * img_h) for x, y in coords]
            if len(poly) >= 3:
                polygons.append(poly)
    return polygons


def polygons_to_mask(polygons, img_w: int, img_h: int) -> np.ndarray:
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon(poly, fill=255)
    return np.array(mask, dtype=np.float32) / 255.0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


# ============================================================================
# Model Loading
# ============================================================================
def load_model(model_repo: str, device: str = "cuda"):
    """Load trained LISA model from HF repo.
    
    Uses snapshot_download to get all files into a clean local directory,
    avoiding HF cache path issues with PeftModel.from_pretrained().
    """
    from huggingface_hub import snapshot_download

    logger.info(f"Loading model from {model_repo}...")

    # Download entire repo to a clean local directory
    local_dir = snapshot_download(model_repo)
    logger.info(f"Model files downloaded to: {local_dir}")

    # Load config
    config_path = os.path.join(local_dir, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    vlm_model_id = config["vlm_model_id"]
    sam_model_id = config["sam_model_id"]
    vlm_hidden_size = config["vlm_hidden_size"]
    sam_prompt_dim = config["sam_prompt_dim"]
    seg_token = config["seg_token"]

    # Processor — try loading saved tokenizer first, fallback to base + re-add token
    tokenizer_dir = os.path.join(local_dir, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        try:
            processor = AutoProcessor.from_pretrained(tokenizer_dir, trust_remote_code=True)
            seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)
            logger.info(f"Loaded tokenizer from saved dir, seg_token_id={seg_token_id}")
        except Exception as e:
            logger.warning(f"Failed to load saved tokenizer: {e}, falling back to base model")
            processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
            processor.tokenizer.add_tokens([seg_token], special_tokens=True)
            seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)
    else:
        processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
        processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        seg_token_id = processor.tokenizer.convert_tokens_to_ids(seg_token)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # VLM + LoRA
    dtype = torch.float16 if device != "cpu" else torch.float32
    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
        vlm_model_id, torch_dtype=dtype, device_map=device,
    )
    vlm.resize_token_embeddings(len(processor.tokenizer))

    # Load LoRA from local path — this avoids the HF cache path validation issue
    lora_dir = os.path.join(local_dir, "vlm_lora")
    logger.info(f"Loading LoRA adapter from: {lora_dir}")
    vlm = PeftModel.from_pretrained(vlm, lora_dir)
    vlm.eval()

    # SAM
    sam = SamModel.from_pretrained(sam_model_id, torch_dtype=torch.float32).to(device)
    sam_processor = SamProcessor.from_pretrained(sam_model_id)

    # Load fine-tuned mask decoder
    decoder_path = os.path.join(local_dir, "sam_mask_decoder.pt")
    sam.mask_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    sam.eval()

    # Seg projector
    seg_projector = SegProjectionMLP(in_dim=vlm_hidden_size, out_dim=sam_prompt_dim).to(device)
    proj_path = os.path.join(local_dir, "seg_projector.pt")
    seg_projector.load_state_dict(torch.load(proj_path, map_location=device))
    seg_projector.eval()

    logger.info(f"Model loaded: VLM={vlm_model_id}, SAM={sam_model_id}")

    return {
        "vlm": vlm, "sam": sam, "sam_processor": sam_processor,
        "seg_projector": seg_projector, "processor": processor,
        "seg_token_id": seg_token_id, "seg_token": seg_token,
        "device": device, "config": config,
    }


# ============================================================================
# Inference Functions
# ============================================================================
@torch.no_grad()
def run_text_inference(model_dict, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
    processor = model_dict["processor"]
    vlm = model_dict["vlm"]
    device = model_dict["device"]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    output_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


@torch.no_grad()
def run_seg_inference(model_dict, image: Image.Image, prompt: str) -> Tuple[str, Optional[np.ndarray]]:
    processor = model_dict["processor"]
    vlm = model_dict["vlm"]
    sam = model_dict["sam"]
    sam_processor = model_dict["sam_processor"]
    seg_projector = model_dict["seg_projector"]
    seg_token_id = model_dict["seg_token_id"]
    device = model_dict["device"]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    output = vlm.generate(
        **inputs, max_new_tokens=128, do_sample=False,
        output_hidden_states=True, return_dict_in_generate=True,
    )

    generated_ids = output.sequences[0]
    input_len = inputs["input_ids"].shape[1]
    text_response = processor.tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True).strip()

    seg_positions = (generated_ids == seg_token_id).nonzero(as_tuple=True)[0]
    mask_np = None

    if len(seg_positions) > 0:
        seg_step_idx = seg_positions[0].item() - input_len
        if 0 <= seg_step_idx < len(output.hidden_states):
            h_seg = output.hidden_states[seg_step_idx][-1][:, -1, :]
            sparse_prompt = seg_projector(h_seg.float()).unsqueeze(1).unsqueeze(1)

            sam_inputs = sam_processor(images=[image], return_tensors="pt").to(device)
            sam_emb = sam.get_image_embeddings(sam_inputs["pixel_values"])
            img_pos = sam.get_image_wide_positional_embeddings()
            _, dense_emb = sam.prompt_encoder(
                input_points=None, input_labels=None,
                input_boxes=None, input_masks=None,
            )

            mask_preds, _ = sam.mask_decoder(
                image_embeddings=sam_emb,
                image_positional_embeddings=img_pos,
                sparse_prompt_embeddings=sparse_prompt,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            mask_np = torch.sigmoid(mask_preds.squeeze()).cpu().numpy()

    return text_response, mask_np


# ============================================================================
# Sampling & Evaluation
# ============================================================================
def sample_from_dataset(split: str, samples_per_task: int, seed: int = 42):
    """Load dataset and sample N examples per task type."""
    logger.info(f"Loading dataset {DATASET_ID} split={split}...")
    ds = load_dataset(DATASET_ID, split=split)

    # Group indices by task
    task_indices = defaultdict(list)
    for i in range(len(ds)):
        task_indices[ds[i]["task"]].append(i)

    random.seed(seed)
    selected = []
    for task, indices in sorted(task_indices.items()):
        k = min(samples_per_task, len(indices))
        chosen = random.sample(indices, k)
        for idx in chosen:
            selected.append((idx, ds[idx]))
        logger.info(f"  {task}: {len(indices)} total, sampled {k}")

    logger.info(f"Total samples: {len(selected)}")
    return selected


def run_evaluation(model_dict, samples, output_dir: str):
    """Run inference on all samples, save images and results."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    results = []

    for i, (idx, row) in enumerate(samples):
        task = row["task"]
        image = row["image"]
        conversations = json.loads(row["conversations"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        if image is None:
            continue
        if image.mode != "RGB":
            image = image.convert("RGB")

        user_prompt = conversations[0]["content"] if conversations else ""
        expected_answer = conversations[1]["content"] if len(conversations) > 1 else ""
        is_seg = task in SEGMENTATION_TASKS

        sample_id = f"sample_{i:03d}"
        logger.info(f"[{i+1}/{len(samples)}] {sample_id} — task={task}")

        # Save input image
        input_path = os.path.join(img_dir, f"{sample_id}_input.png")
        image.save(input_path)

        result = {
            "sample_id": sample_id,
            "dataset_idx": idx,
            "task": task,
            "source": row.get("source", ""),
            "prompt": user_prompt,
            "expected": expected_answer[:500],  # Truncate long polygon text
            "input_image": f"images/{sample_id}_input.png",
        }

        try:
            if is_seg:
                # Run segmentation inference
                text_out, pred_mask = run_seg_inference(model_dict, image, user_prompt)
                result["model_output"] = text_out
                result["seg_token_generated"] = pred_mask is not None

                # Save predicted mask
                if pred_mask is not None:
                    pred_bin = (pred_mask > 0.5).astype(np.uint8) * 255
                    pred_img = Image.fromarray(pred_bin).resize(image.size, Image.NEAREST)
                    pred_path = os.path.join(img_dir, f"{sample_id}_pred_mask.png")
                    pred_img.save(pred_path)
                    result["pred_mask"] = f"images/{sample_id}_pred_mask.png"

                # Generate GT mask from polygon text
                img_w = metadata.get("image_width", image.size[0])
                img_h = metadata.get("image_height", image.size[1])
                polygons = parse_polygons_from_text(expected_answer, img_w, img_h)
                if polygons:
                    gt_mask = polygons_to_mask(polygons, img_w, img_h)
                    gt_bin = (gt_mask > 0.5).astype(np.uint8) * 255
                    gt_img = Image.fromarray(gt_bin).resize(image.size, Image.NEAREST)
                    gt_path = os.path.join(img_dir, f"{sample_id}_gt_mask.png")
                    gt_img.save(gt_path)
                    result["gt_mask"] = f"images/{sample_id}_gt_mask.png"
                    result["num_gt_polygons"] = len(polygons)

                    # Compute IoU if both masks exist
                    if pred_mask is not None:
                        gt_resized = np.array(gt_img.resize(
                            (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST
                        ), dtype=np.float32) / 255.0
                        iou = compute_iou(pred_mask, gt_resized)
                        result["iou"] = round(iou, 4)
            else:
                # Run text inference
                text_out = run_text_inference(model_dict, image, user_prompt)
                result["model_output"] = text_out

        except Exception as e:
            logger.warning(f"  Error on {sample_id}: {e}")
            result["error"] = str(e)

        results.append(result)

    return results


# ============================================================================
# Report Generation
# ============================================================================
def generate_report(results, output_dir: str, model_repo: str, split: str):
    """Generate REPORT.md with structured results."""

    report_lines = []
    r = report_lines.append

    r(f"# 🛰️ Satellite Omni LISA — Evaluation Report")
    r(f"")
    r(f"**Model:** [`{model_repo}`](https://huggingface.co/{model_repo})")
    r(f"**Dataset:** [`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID})")
    r(f"**Split:** `{split}`")
    r(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r(f"**Total Samples:** {len(results)}")
    r(f"")

    # ---- Summary Table ----
    task_stats = defaultdict(lambda: {"count": 0, "errors": 0, "ious": []})
    for res in results:
        t = res["task"]
        task_stats[t]["count"] += 1
        if "error" in res:
            task_stats[t]["errors"] += 1
        if "iou" in res:
            task_stats[t]["ious"].append(res["iou"])

    r(f"## Summary")
    r(f"")
    r(f"| Task | Samples | Errors | Mean IoU |")
    r(f"|------|---------|--------|----------|")
    for task in sorted(task_stats):
        s = task_stats[task]
        iou_str = f"{np.mean(s['ious']):.4f}" if s["ious"] else "N/A"
        r(f"| {task} | {s['count']} | {s['errors']} | {iou_str} |")
    r(f"")

    # ---- Per-Task Sections ----
    results_by_task = defaultdict(list)
    for res in results:
        results_by_task[res["task"]].append(res)

    for task in sorted(results_by_task):
        task_results = results_by_task[task]
        is_seg = task in SEGMENTATION_TASKS
        emoji = "🔲" if is_seg else "💬"

        r(f"---")
        r(f"## {emoji} {task}")
        r(f"")

        for res in task_results:
            sid = res["sample_id"]
            r(f"### {sid}")
            r(f"")
            r(f"**Source:** {res.get('source', 'N/A')}  ")
            r(f"**Prompt:** {res['prompt'][:200]}")
            r(f"")

            # Input image
            r(f"**Input Image:**")
            r(f"")
            r(f"![input]({res['input_image']})")
            r(f"")

            if "error" in res:
                r(f"⚠️ **Error:** `{res['error']}`")
                r(f"")
                continue

            if is_seg:
                r(f"**Model Text Output:** {res.get('model_output', 'N/A')}")
                r(f"")
                r(f"**`<SEG>` token generated:** {'✅ Yes' if res.get('seg_token_generated') else '❌ No'}")
                r(f"")

                # Side-by-side masks
                has_gt = "gt_mask" in res
                has_pred = "pred_mask" in res

                if has_gt or has_pred:
                    r(f"| Ground Truth Mask | Predicted Mask |")
                    r(f"|:-----------------:|:--------------:|")
                    gt_cell = f"![gt]({res['gt_mask']})" if has_gt else "N/A"
                    pred_cell = f"![pred]({res['pred_mask']})" if has_pred else "No mask generated"
                    r(f"| {gt_cell} | {pred_cell} |")
                    r(f"")

                if "iou" in res:
                    iou_val = res["iou"]
                    iou_bar = "🟩" * int(iou_val * 10) + "⬜" * (10 - int(iou_val * 10))
                    r(f"**IoU:** `{iou_val:.4f}` {iou_bar}")
                    r(f"")

                if "num_gt_polygons" in res:
                    r(f"**GT polygons:** {res['num_gt_polygons']}")
                    r(f"")
            else:
                # Text task — show expected vs actual
                r(f"| | Content |")
                r(f"|---|---|")
                expected_clean = res.get("expected", "").replace("\n", " ").replace("|", "\\|")[:300]
                actual_clean = res.get("model_output", "").replace("\n", " ").replace("|", "\\|")[:300]
                r(f"| **Expected** | {expected_clean} |")
                r(f"| **Model Output** | {actual_clean} |")
                r(f"")

            r(f"")

    # ---- Footer ----
    r(f"---")
    r(f"*Generated by `infer_dataset.py` — "
      f"[Model Repo](https://huggingface.co/{model_repo}) • "
      f"[Dataset](https://huggingface.co/datasets/{DATASET_ID})*")

    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Satellite Omni LISA on dataset samples")
    parser.add_argument("--model_repo", type=str, default="rahuldshetty/satellite-omni-lisa",
                        help="HF repo ID for trained model")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split: train, validation, test")
    parser.add_argument("--samples_per_task", type=int, default=3,
                        help="Number of samples per task type")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="Output directory for results and report")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Sample from dataset
    samples = sample_from_dataset(args.split, args.samples_per_task, args.seed)

    # 2. Load model
    model_dict = load_model(args.model_repo, device=args.device)

    # 3. Run evaluation
    results = run_evaluation(model_dict, samples, args.output_dir)

    # 4. Save raw results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results JSON saved to {results_path}")

    # 5. Generate report
    report_path = generate_report(results, args.output_dir, args.model_repo, args.split)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Report:      {report_path}")
    print(f"  Results:     {results_path}")
    print(f"  Samples:     {len(results)}")

    seg_ious = [r["iou"] for r in results if "iou" in r]
    if seg_ious:
        print(f"  Mean IoU:    {np.mean(seg_ious):.4f}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  Errors:      {len(errors)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
