---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-0.8B
  - Zigeng/SlimSAM-uniform-77
datasets:
  - rahuldshetty/satellite-multitask-omni
tags:
  - satellite
  - remote-sensing
  - segmentation
  - multimodal
  - vision-language
  - LISA
  - image-generation
pipeline_tag: image-text-to-text
---

# 🛰️ Satellite Omni LISA

**A multi-modal satellite model that understands AND generates images.**

| Component | Model | Role |
|-----------|-------|------|
| VLM | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Text + vision understanding |
| Mask Decoder | [SlimSAM-77](https://huggingface.co/Zigeng/SlimSAM-uniform-77) | Image (mask) generation — 10x lighter than SAM-ViT-B |
| Connector | MLP Projection (1024→256) | Bridges VLM → SAM |

## Architecture

Based on [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692), adapted for satellite/remote sensing with multi-task training. Uses [SlimSAM](https://arxiv.org/abs/2312.05284) (9.7M params, 77% pruned) instead of SAM-ViT-B (93.7M) for efficiency.

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT                                 │
│  Satellite Image + Text Prompt                          │
│  e.g., "Segment the buildings in this aerial image"     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             Qwen3.5-0.8B VLM (LoRA fine-tuned)          │
│  ┌────────────┐  ┌───────────────────────────────────┐  │
│  │ ViT Vision │  │ Language Model (24 layers, 1024d)  │  │
│  │  Encoder   │──│ Linear + Full Attention layers     │  │
│  │  (768d)    │  │ MRoPE positional encoding          │  │
│  └────────────┘  └──────────┬────────────────────────┘  │
└─────────────────────────────┼───────────────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                              │
     ┌────────▼─────────┐          ┌─────────▼──────────┐
     │   Text Output     │          │  <SEG> Token        │
     │                   │          │  Hidden State       │
     │  "I identified    │          │  (1024-dim)         │
     │   5 buildings"    │          └─────────┬──────────┘
     └───────────────────┘                    │
                                    ┌─────────▼──────────┐
                                    │  MLP Projection     │
                                    │  1024 → 256         │
                                    │  (GELU activation)  │
                                    └─────────┬──────────┘
                                              │
     ┌────────────────────────────────────────▼──────────┐
     │            SlimSAM-77                              │
     │  ┌─────────────┐  ┌────────────────────────────┐  │
     │  │ ViT Encoder  │  │ Mask Decoder (fine-tuned)  │  │
     │  │ (frozen)     │──│ Cross-attention + MLP      │  │
     │  │ ~5.7M params │  │ ~4M params                 │  │
     │  └─────────────┘  └──────────┬─────────────────┘  │
     └───────────────────────────────┼───────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Binary Mask Image  │
                           │  (256×256 → resize) │
                           └─────────────────────┘
```

## Key Innovation

The model handles **two output modalities** from a single VLM:

1. **Text output**: Standard autoregressive text generation for captioning, VQA, classification
2. **Image output**: When the model generates a special `<SEG>` token, its hidden representation is projected through an MLP and used as a prompt for SlimSAM's mask decoder to generate a segmentation mask **image**

This means the model learns **when** to generate an image (by outputting `<SEG>`) and **what** image to generate (the hidden state encodes the full visual context).

## Tasks

### Text Output Tasks
| Task | Example |
|------|---------|
| Image Captioning | "Describe this satellite image" → "Green trees surround industrial buildings near a river..." |
| Visual QA | "What type of land cover is shown?" → "Agricultural farmland" |
| Scene Classification | "Classify this satellite image" → "Industrial Buildings" |
| Object Detection | "What objects are visible?" → "3 buildings and 2 roads" |
| Flood Assessment | "Is there flooding visible?" → "Yes, 2 flooded buildings visible" |

### Image Output Tasks
| Task | Example |
|------|---------|
| Building Segmentation | "Segment the buildings" → Text + Binary mask image |
| Instance Segmentation | "Detect and segment all structures" → Text + Per-instance masks |

## Training Details

### Loss Function
```
L = L_text + L_mask
L_text = CrossEntropy(predicted_tokens, target_tokens)
L_mask = 2.0 * BCE(predicted_mask, gt_mask) + 0.5 * DICE(predicted_mask, gt_mask)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Effective batch size | 16 (1×16 or 2×8 grad accum, auto-detected) |
| Learning rate | 2e-4 |
| Scheduler | Cosine with 5% warmup |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Max sequence length | 512 |
| Precision | FP16 (T4/Colab) or BF16 (A10G+) — auto-detected |

### Parameter Budget
| Component | Params | Trainable? |
|-----------|--------|-----------|
| Qwen3.5-0.8B VLM | ~873M | LoRA only (~5M) |
| SlimSAM-77 Encoder | ~5.7M | ❄️ Frozen |
| SlimSAM Mask Decoder | ~4M | ✅ Yes |
| Seg Projection MLP | ~1M | ✅ Yes |
| **Total trainable** | **~10M** | **~1% of total** |

## Dataset

Trained on [rahuldshetty/satellite-multitask-omni](https://huggingface.co/datasets/rahuldshetty/satellite-multitask-omni):
- ~20K training samples across 9 task types
- Sources: EuroSAT, RSICD, PatternNet, FloodNet, building segmentation datasets
- Segmentation masks from normalized polygon coordinates (0-1000 range)

## Usage

### Installation
```bash
# Qwen3.5 requires transformers from git (>= 4.57.0)
pip install "transformers @ git+https://github.com/huggingface/transformers.git"
pip install torch torchvision peft datasets Pillow numpy huggingface_hub

# Or use the setup script:
bash setup.sh
```

### Single-Image Inference
```python
from inference import load_model, generate_text, generate_segmentation
from PIL import Image

# Load model
model = load_model("./model_dir", device="cuda")

# Text task
image = Image.open("satellite.jpg")
response = generate_text(model, image, "Describe this satellite image")
print(response)

# Segmentation task
text, mask = generate_segmentation(model, image, "Segment the buildings")
print(text)
mask.save("building_mask.png")
```

### CLI Inference
```bash
# Text generation
python inference.py --model_dir ./model --image sat.jpg --prompt "What land cover type?"

# Segmentation (generates mask image)
python inference.py --model_dir ./model --image sat.jpg --prompt "Segment buildings" --output_mask mask.png
```

### Dataset Evaluation

Run inference on sampled dataset examples across all task types and generate a structured report:

```bash
# Default: 3 samples per task from validation split
python infer_dataset.py --model_repo rahuldshetty/satellite-omni-lisa

# More samples, custom output
python infer_dataset.py --model_repo rahuldshetty/satellite-omni-lisa \
    --samples_per_task 5 \
    --split test \
    --output_dir ./my_eval

# Force CPU (slower but works without GPU)
python infer_dataset.py --device cpu --samples_per_task 2
```

This produces:
```
eval_output/
├── REPORT.md              ← Markdown report with all results
├── results.json           ← Machine-readable results
└── images/
    ├── sample_000_input.png
    ├── sample_003_gt_mask.png
    ├── sample_003_pred_mask.png
    └── ...
```

The **REPORT.md** includes:
- **Summary table** — per-task sample counts, error counts, mean IoU for segmentation
- **Per-task sections** — each sample shows:
  - Input satellite image
  - Text tasks: expected vs model output in a comparison table
  - Segmentation tasks: GT mask vs predicted mask side-by-side with IoU score

### Training
```bash
# Auto-detects GPU, sets batch size, bf16/fp16 automatically
python train.py

# Override settings via environment variables
BATCH_SIZE=1 GRAD_ACCUM=16 MAX_SAMPLES=100 python train.py
```

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BATCH_SIZE` | Auto (1 for T4, 2 for A10G+) | Per-device batch size |
| `GRAD_ACCUM` | `16 // BATCH_SIZE` | Gradient accumulation steps |
| `MAX_SAMPLES` | 0 (all) | Limit dataset size for debugging |
| `NUM_WORKERS` | 0 | Dataloader workers (0 = main process, saves RAM) |

## Files

| File | Description |
|------|-------------|
| `train.py` | Full training script with auto hardware detection |
| `inference.py` | Single-image inference (text + segmentation) |
| `infer_dataset.py` | Dataset evaluation — samples all tasks, generates REPORT.md |
| `setup.sh` | Auto-installs transformers from git if needed |
| `requirements.txt` | Pip dependencies with version pins |
| `vlm_lora/` | LoRA adapter weights for Qwen3.5-0.8B |
| `seg_projector.pt` | MLP projection weights (1024→256) |
| `sam_mask_decoder.pt` | Fine-tuned SlimSAM mask decoder weights |
| `tokenizer/` | Tokenizer with added `<SEG>` token |
| `model_config.json` | Model configuration (model IDs, dimensions, tokens) |

## References

- [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692) (CVPR 2024)
- [SlimSAM: 0.1% Data Makes Segment Anything Slim](https://arxiv.org/abs/2312.05284)
- [LISAT: Remote Sensing Adaptation](https://arxiv.org/abs/2505.02829)
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [SAM: Segment Anything](https://arxiv.org/abs/2304.02643)
