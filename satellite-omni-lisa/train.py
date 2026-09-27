"""
LISA-style Satellite Multi-Task Omni Model
==========================================
Architecture: Qwen3.5-0.8B (VLM) + SlimSAM (mask decoder)
- Text tasks: standard VLM fine-tuning (captioning, VQA, classification)
- Segmentation tasks: VLM outputs <SEG> token → MLP projection → SlimSAM mask decoder → binary mask

Based on LISA (arxiv:2308.00692) paradigm adapted for satellite imagery.
Uses SlimSAM (arxiv:2312.05284) instead of SAM-ViT-B — 10x smaller (9.7M vs 93.7M params), same API.

Requirements:
    Qwen3.5 support requires transformers >= 4.57.0 (installed from git main if not yet released on PyPI).
    See requirements.txt for full dependencies or run:

    pip install -r requirements.txt
    python train.py

Hardware: A10G (24GB VRAM) recommended. T4/Colab (16GB) supported with automatic settings.
VRAM estimate: ~6-8GB (VLM bf16 ~1.7GB + SlimSAM fp32 ~0.04GB + encoder fwd ~0.5GB + grads/optim ~4GB)

Environment variables (optional):
    BATCH_SIZE=2          Override per-device batch size (default: 2 for T4, 4 for A10G+)
    GRAD_ACCUM=8          Override gradient accumulation steps (default: auto)
    MAX_SAMPLES=0         Limit dataset size for debugging (0 = use all)
    NUM_WORKERS=0         Dataloader workers (default: 0 for Colab compatibility)
"""

import os
import re
import json
import math
import random
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image, ImageDraw
from datasets import load_dataset

import transformers
from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen3_5ForConditionalGeneration,
    SamModel,
    SamProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

import trackio

# ============================================================================
# Configuration
# ============================================================================
VLM_MODEL_ID = "Qwen/Qwen3.5-0.8B"
SAM_MODEL_ID = "Zigeng/SlimSAM-uniform-77"
OUTPUT_MODEL_ID = "rahuldshetty/satellite-omni-lisa"
DATASET_ID = "rahuldshetty/satellite-multitask-omni"

SEG_TOKEN = "<SEG>"
DEFAULT_IMAGE_SIZE = 256
SAM_IMAGE_SIZE = 1024

SEGMENTATION_TASKS = {"instance_segmentation"}
TEXT_TASKS = {
    "image_captioning", "visual_question_answering", "scene_classification",
    "scene_classification_vqa", "captioning_vqa", "flood_vqa",
    "multi_turn_conversation", "object_detection"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Polygon to Mask Conversion
# ============================================================================
def parse_polygons_from_text(text: str, img_w: int, img_h: int) -> List[List[Tuple[float, float]]]:
    polygons = []
    pattern = r'polygon\s*\[(.*?)\]'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        coords = re.findall(r'\((\d+),\s*(\d+)\)', match)
        if coords:
            poly = []
            for x_str, y_str in coords:
                x = int(x_str) / 1000.0 * img_w
                y = int(y_str) / 1000.0 * img_h
                poly.append((x, y))
            if len(poly) >= 3:
                polygons.append(poly)
    return polygons


def polygons_to_mask(polygons: List[List[Tuple[float, float]]], img_w: int, img_h: int) -> np.ndarray:
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon(poly, fill=255)
    return np.array(mask, dtype=np.float32) / 255.0


# ============================================================================
# Model Architecture
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


class SatelliteOmniLISA(nn.Module):
    def __init__(self, vlm_model, sam_model, seg_token_id: int, vlm_hidden_size: int = 1024):
        super().__init__()
        self.vlm = vlm_model
        self.sam = sam_model
        self.seg_token_id = seg_token_id
        self.config = vlm_model.config

        sam_prompt_dim = sam_model.config.mask_decoder_config.hidden_size
        self.seg_projector = SegProjectionMLP(in_dim=vlm_hidden_size, out_dim=sam_prompt_dim)

        for param in self.sam.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.vlm.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.vlm, 'is_gradient_checkpointing', False)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None,
                mm_token_type_ids=None, labels=None, sam_pixel_values=None,
                gt_masks=None, has_seg_mask=None, **kwargs):
        vlm_outputs = self.vlm(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids, labels=labels,
            output_hidden_states=True,
        )

        text_loss = vlm_outputs.loss if vlm_outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        mask_loss = torch.tensor(0.0, device=input_ids.device, dtype=text_loss.dtype)
        pred_masks = None

        if (has_seg_mask is not None and has_seg_mask.any()
                and sam_pixel_values is not None and gt_masks is not None):
            seg_indices = has_seg_mask.nonzero(as_tuple=True)[0]
            hidden_states = vlm_outputs.hidden_states[-1]

            seg_hiddens, valid_seg_indices = [], []
            for idx in seg_indices:
                seg_positions = (input_ids[idx] == self.seg_token_id).nonzero(as_tuple=True)[0]
                if len(seg_positions) > 0:
                    seg_hiddens.append(hidden_states[idx, seg_positions[0], :])
                    valid_seg_indices.append(idx)

            if len(seg_hiddens) > 0:
                seg_hiddens = torch.stack(seg_hiddens, dim=0)
                valid_seg_indices = torch.stack(valid_seg_indices)
                N_seg = len(seg_hiddens)

                sparse_prompts = self.seg_projector(seg_hiddens.float()).unsqueeze(1).unsqueeze(1)

                seg_sam_pixels = sam_pixel_values[valid_seg_indices]
                with torch.no_grad():
                    sam_image_embeddings = self.sam.get_image_embeddings(seg_sam_pixels)
                image_pos_embeddings = self.sam.get_image_wide_positional_embeddings().expand(N_seg, -1, -1, -1)
                with torch.no_grad():
                    _, dense_embeddings = self.sam.prompt_encoder(
                        input_points=None, input_labels=None, input_boxes=None, input_masks=None)
                dense_embeddings = dense_embeddings.expand(N_seg, -1, -1, -1)

                mask_preds, _ = self.sam.mask_decoder(
                    image_embeddings=sam_image_embeddings,
                    image_positional_embeddings=image_pos_embeddings,
                    sparse_prompt_embeddings=sparse_prompts,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                mask_preds = mask_preds.squeeze(1).squeeze(1)

                seg_gt_masks = gt_masks[valid_seg_indices]
                pred_h, pred_w = mask_preds.shape[-2], mask_preds.shape[-1]
                if seg_gt_masks.shape[-2:] != (pred_h, pred_w):
                    seg_gt_masks = F.interpolate(
                        seg_gt_masks.unsqueeze(1).float(), size=(pred_h, pred_w),
                        mode='bilinear', align_corners=False).squeeze(1)
                seg_gt_masks = seg_gt_masks.to(mask_preds.dtype)

                bce_loss = F.binary_cross_entropy_with_logits(mask_preds, seg_gt_masks)
                pred_probs = torch.sigmoid(mask_preds)
                pred_flat = pred_probs.reshape(N_seg, -1)
                gt_flat = seg_gt_masks.reshape(N_seg, -1)
                intersection = (pred_flat * gt_flat).sum(dim=1)
                dice = 1 - (2 * intersection + 1) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1)
                mask_loss = 2.0 * bce_loss + 0.5 * dice.mean()
                pred_masks = mask_preds

        return {"loss": text_loss + mask_loss, "text_loss": text_loss.detach(),
                "mask_loss": mask_loss.detach(), "pred_masks": pred_masks}


# ============================================================================
# Dataset
# ============================================================================
class SatelliteOmniDataset(Dataset):
    def __init__(self, hf_dataset, processor, sam_processor, seg_token_id, max_length=512, split="train"):
        self.data = hf_dataset
        self.processor = processor
        self.sam_processor = sam_processor
        self.seg_token_id = seg_token_id
        self.max_length = max_length
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        logger.info(f"Loaded {len(self.data)} examples for {split} (lazy)")

    def __len__(self):
        return len(self.data)

    def _find_assistant_start(self, input_ids):
        ids = input_ids.tolist()
        for i in range(len(ids) - 2):
            if ids[i] == self.im_start_id:
                decoded = self.processor.tokenizer.decode(ids[i+1:i+5], skip_special_tokens=False)
                if "assistant" in decoded:
                    return i + 3
        return 0

    def __getitem__(self, idx):
        try:
            return self._get_item_impl(idx)
        except Exception as e:
            logger.warning(f"Error sample {idx}: {e}")
            return self._get_item_impl((idx + 1) % len(self))

    def _get_item_impl(self, idx):
        row = self.data[idx]
        task, image = row["task"], row["image"]
        conversations = json.loads(row["conversations"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        is_seg = task in SEGMENTATION_TASKS

        if image is None:
            return self._get_item_impl((idx + 1) % len(self))
        if image.mode != "RGB":
            image = image.convert("RGB")

        if is_seg:
            user_content = conversations[0]["content"] if conversations else "Segment the objects."
            num_objects = metadata.get("num_objects", "several")
            assistant_text = f"I identified {num_objects} building instance(s) in this image. {SEG_TOKEN}"
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_content}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ]
        else:
            user_msg = conversations[0]["content"] if conversations else ""
            assistant_msg = conversations[1]["content"] if len(conversations) > 1 else ""
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_msg}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
            ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        vlm_inputs = self.processor(text=[text], images=[image], return_tensors="pt",
                                     padding="max_length", max_length=self.max_length, truncation=True)

        input_ids = vlm_inputs["input_ids"].squeeze(0)
        attention_mask = vlm_inputs["attention_mask"].squeeze(0)
        pixel_values = vlm_inputs["pixel_values"]
        image_grid_thw = vlm_inputs["image_grid_thw"].squeeze(0)

        labels = input_ids.clone()
        labels[:self._find_assistant_start(input_ids)] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask,
                  "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
                  "labels": labels, "has_seg_mask": torch.tensor(is_seg, dtype=torch.bool)}

        if "mm_token_type_ids" in vlm_inputs:
            result["mm_token_type_ids"] = vlm_inputs["mm_token_type_ids"].squeeze(0)

        if is_seg:
            sam_inputs = self.sam_processor(images=[image], return_tensors="pt")
            result["sam_pixel_values"] = sam_inputs["pixel_values"].squeeze(0)
            img_w = metadata.get("image_width", image.size[0])
            img_h = metadata.get("image_height", image.size[1])
            orig_text = conversations[1]["content"] if len(conversations) > 1 else ""
            polygons = parse_polygons_from_text(orig_text, img_w, img_h)
            if polygons:
                mask = polygons_to_mask(polygons, img_w, img_h)
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.BILINEAR)
                gt_mask = np.array(mask_pil, dtype=np.float32) / 255.0
            else:
                gt_mask = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.float32)
            result["gt_masks"] = torch.tensor(gt_mask, dtype=torch.float32)
        else:
            result["sam_pixel_values"] = torch.zeros(3, 64, 64, dtype=torch.float32)
            result["gt_masks"] = torch.zeros(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, dtype=torch.float32)

        return result


# ============================================================================
# Custom Collator
# ============================================================================
class OmniCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "has_seg_mask": torch.stack([f["has_seg_mask"] for f in features]),
            "gt_masks": torch.stack([f["gt_masks"] for f in features]),
        }

        if any(f["has_seg_mask"].item() for f in features):
            sam_pvs = []
            for f in features:
                pv = f["sam_pixel_values"]
                if pv.shape[-1] != SAM_IMAGE_SIZE:
                    pv = torch.zeros(3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE, dtype=pv.dtype)
                sam_pvs.append(pv)
            batch["sam_pixel_values"] = torch.stack(sam_pvs)
        else:
            batch["sam_pixel_values"] = torch.zeros(len(features), 3, 64, 64, dtype=torch.float32)

        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])
        if "mm_token_type_ids" in features[0]:
            batch["mm_token_type_ids"] = torch.stack([f["mm_token_type_ids"] for f in features])
        return batch


# ============================================================================
# Hub Upload Callback
# ============================================================================
class PushComponentsCallback(TrainerCallback):
    """Push all model components to Hub on every checkpoint save."""

    def __init__(self, model, processor, sam_model, seg_token_id, vlm_hidden_size, hub_model_id):
        self.model = model
        self.processor = processor
        self.sam_model = sam_model
        self.seg_token_id = seg_token_id
        self.vlm_hidden_size = vlm_hidden_size
        self.hub_model_id = hub_model_id

    def on_save(self, args, state, control, **kwargs):
        logger.info(f"📤 Pushing to Hub (step {state.global_step})...")
        try:
            from huggingface_hub import HfApi
            import shutil
            api = HfApi()

            save_dir = os.path.join(args.output_dir, "hub_upload_tmp")
            os.makedirs(save_dir, exist_ok=True)

            self.model.vlm.save_pretrained(os.path.join(save_dir, "vlm_lora"))
            torch.save(self.model.seg_projector.state_dict(), os.path.join(save_dir, "seg_projector.pt"))
            torch.save(self.model.sam.mask_decoder.state_dict(), os.path.join(save_dir, "sam_mask_decoder.pt"))

            config_info = {
                "vlm_model_id": VLM_MODEL_ID, "sam_model_id": SAM_MODEL_ID,
                "seg_token": SEG_TOKEN, "seg_token_id": self.seg_token_id,
                "vlm_hidden_size": self.vlm_hidden_size,
                "sam_prompt_dim": self.sam_model.config.mask_decoder_config.hidden_size,
                "lora_r": 16, "lora_alpha": 32,
                "architecture": "LISA (Qwen3.5-VL-0.8B + SlimSAM-77)",
                "training_step": state.global_step,
                "training_epoch": round(state.epoch, 4) if state.epoch else 0,
            }
            with open(os.path.join(save_dir, "model_config.json"), "w") as f:
                json.dump(config_info, f, indent=2)

            self.processor.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

            api.upload_folder(repo_id=self.hub_model_id, folder_path=save_dir,
                              commit_message=f"Checkpoint step {state.global_step} (epoch {state.epoch:.2f})")
            logger.info(f"✅ Pushed to https://huggingface.co/{self.hub_model_id} (step {state.global_step})")
            shutil.rmtree(save_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"⚠️ Hub push failed (non-fatal): {e}")


# ============================================================================
# Custom Trainer — overrides _save to avoid safetensors shared tensor crash
# ============================================================================
class OmniTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override Trainer._save to save our composite model properly.
        
        The default Trainer._save tries to serialize the entire nn.Module via safetensors,
        which crashes on shared/tied tensors (VLM weight tying + SAM shared embeddings).
        Instead, we save each component separately using its own save method.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving checkpoint to {output_dir}")

        model = self.model

        # Save LoRA adapter (handles weight tying correctly)
        model.vlm.save_pretrained(os.path.join(output_dir, "vlm_lora"))

        # Save custom components as plain torch files (no safetensors issues)
        torch.save(model.seg_projector.state_dict(), os.path.join(output_dir, "seg_projector.pt"))
        torch.save(model.sam.mask_decoder.state_dict(), os.path.join(output_dir, "sam_mask_decoder.pt"))

        # Save trainer state
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"), labels=inputs["labels"],
            sam_pixel_values=inputs.get("sam_pixel_values"), gt_masks=inputs.get("gt_masks"),
            has_seg_mask=inputs.get("has_seg_mask"),
        )
        loss = outputs["loss"]
        if self.state.global_step % 10 == 0:
            self.log({"text_loss": outputs["text_loss"].item(), "mask_loss": outputs["mask_loss"].item()})
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main
# ============================================================================
def main():
    gpu_name = ""
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)" if gpu_name else "No GPU detected")

    use_bf16 = gpu_mem_gb >= 20
    use_fp16 = (not use_bf16) and torch.cuda.is_available()

    if int(os.environ.get("BATCH_SIZE", 0)) > 0:
        batch_size = int(os.environ["BATCH_SIZE"])
    elif gpu_mem_gb >= 20:
        batch_size = 4
    else:
        batch_size = 2

    grad_accum = int(os.environ.get("GRAD_ACCUM", max(1, 16 // batch_size)))
    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    max_samples = int(os.environ.get("MAX_SAMPLES", 0))

    logger.info(f"Config: fp16={use_fp16}, bf16={use_bf16}, batch={batch_size}, "
                f"grad_accum={grad_accum}, effective_batch={batch_size * grad_accum}")

    trackio.init(project="satellite-omni-lisa", run="train-v1")
    logger.info("=" * 60)
    logger.info("LISA-style Satellite Multi-Task Omni Model Training")
    logger.info("=" * 60)

    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
    processor.tokenizer.add_tokens([SEG_TOKEN], special_tokens=True)
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    logger.info(f"<SEG> ID: {seg_token_id}")

    vlm_dtype = torch.bfloat16 if use_bf16 else torch.float16
    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(VLM_MODEL_ID, torch_dtype=vlm_dtype, attn_implementation="sdpa")
    vlm.resize_token_embeddings(len(processor.tokenizer))
    vlm = get_peft_model(vlm, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                                          lora_dropout=0.05, task_type="CAUSAL_LM"))
    vlm.print_trainable_parameters()

    sam = SamModel.from_pretrained(SAM_MODEL_ID, torch_dtype=torch.float32)
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    logger.info(f"SlimSAM loaded: vision={sam.config.vision_config.hidden_size}, decoder={sam.config.mask_decoder_config.hidden_size}")

    vlm_hidden_size = vlm.config.text_config.hidden_size
    model = SatelliteOmniLISA(vlm_model=vlm, sam_model=sam, seg_token_id=seg_token_id, vlm_hidden_size=vlm_hidden_size)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_p:,} | Trainable: {train_p:,} ({100*train_p/total_p:.2f}%)")

    dataset = load_dataset(DATASET_ID)
    train_data, val_data = dataset["train"], dataset["validation"]
    if max_samples > 0:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = SatelliteOmniDataset(train_data, processor, sam_processor, seg_token_id, max_length=512, split="train")
    val_dataset = SatelliteOmniDataset(val_data, processor, sam_processor, seg_token_id, max_length=512, split="validation")

    training_args = TrainingArguments(
        output_dir="./output", hub_model_id=OUTPUT_MODEL_ID, push_to_hub=False,
        num_train_epochs=2, per_device_train_batch_size=batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, max_grad_norm=1.0,
        bf16=use_bf16, fp16=use_fp16, tf32=use_bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps", logging_steps=10, logging_first_step=True, disable_tqdm=True,
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        dataloader_num_workers=num_workers, dataloader_pin_memory=(num_workers > 0),
        remove_unused_columns=False, report_to="none",
    )

    push_callback = PushComponentsCallback(
        model=model, processor=processor, sam_model=sam,
        seg_token_id=seg_token_id, vlm_hidden_size=vlm_hidden_size, hub_model_id=OUTPUT_MODEL_ID)

    trainer = OmniTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=OmniCollator(pad_token_id=processor.tokenizer.pad_token_id),
        callbacks=[push_callback],
    )

    trainer.train()

    # Final push
    logger.info("Final push to Hub...")
    save_dir = "./output/final"
    os.makedirs(save_dir, exist_ok=True)
    model.vlm.save_pretrained(os.path.join(save_dir, "vlm_lora"))
    torch.save(model.seg_projector.state_dict(), os.path.join(save_dir, "seg_projector.pt"))
    torch.save(model.sam.mask_decoder.state_dict(), os.path.join(save_dir, "sam_mask_decoder.pt"))
    config_info = {
        "vlm_model_id": VLM_MODEL_ID, "sam_model_id": SAM_MODEL_ID,
        "seg_token": SEG_TOKEN, "seg_token_id": seg_token_id,
        "vlm_hidden_size": vlm_hidden_size,
        "sam_prompt_dim": sam.config.mask_decoder_config.hidden_size,
        "lora_r": 16, "lora_alpha": 32,
        "architecture": "LISA (Qwen3.5-VL-0.8B + SlimSAM-77)",
        "training_complete": True,
    }
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    processor.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=OUTPUT_MODEL_ID, folder_path=save_dir,
                          commit_message="LISA satellite omni — final (training complete)")
    logger.info(f"✅ Done! Model at https://huggingface.co/{OUTPUT_MODEL_ID}")


if __name__ == "__main__":
    main()
