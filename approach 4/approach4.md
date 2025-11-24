# Approach

This describes the multimodal price prediction pipeline implemented in `main.ipynb`, which fuses text, image, and structured features.

## Objective
Predict product price using:
- Text: item_name + bullet points (DeBERTa-v3-large)
- Image: product image (CLIP ViT-L/14)
- Numeric: parsed value and IPQ plus one-hot encoded unit

Target: log1p(price), predictions are expm1 back to price scale.

## Data preparation
- Inputs: `../dataset/train.csv`, `../dataset/test.csv`; images in `../images_train/`, `../images_test/` via `image_link`.
- Parsing:
  - parse_catalog_content → item_name, bullet_points, value (float/NaN), unit (string).
  - extract_ipq → pack quantity from phrases like “pack of X”, “X-pack”, “X count/ct”, “case of X”; default 1.
- Features:
  - clean_text = item_name + " " + bullet_points.
  - Drop: catalog_content, filename, item_name, bullet_points, image_link.
  - Impute value with train median; apply to test.
  - One-hot encode unit (handle_unknown="ignore").
  - Scale numeric ["value","ipq"] via StandardScaler (fit on train; apply to val/test).
  - Train/val split: 90/10, seed=42.
  - Add log_price = log1p(price).

## Tokenization and images
- Text: AutoTokenizer for `microsoft/deberta-v3-large`, max_length=128, truncation/padding.
- Image: CLIPProcessor for `openai/clip-vit-large-patch14` (resize, center-crop, normalize).
- Fallback: blank RGB image if missing/corrupt.

## Dataset
Each sample provides:
- input_ids, attention_mask (text)
- pixel_values (image)
- numeric tensor: scaled value, ipq, and unit_* one-hots
- Training only: log_price and price

## Model
MultiModalRegressor:
- Text branch: DeBERTa-v3-large → mean-pool last_hidden_state → Linear 1024→512.
- Image branch: CLIP ViT-L/14 vision pooler_output → Linear 1024→512.
- Numeric branch: Linear num_features→128.
- Fusion: concat(512+512+128) → MLP head: 1152→512→ReLU→Dropout(0.3)→1 (predict log_price).

## Training
- Device: CUDA if available.
- Effective batch: gradient accumulation.
  - Physical batch: 16
  - GRAD_ACCUM_STEPS: 2 → effective batch 32
- Epochs: 30
- Optimizer: AdamW (lr=1e-5)
- Loss: SmoothL1 on log_price
- Scheduler: linear, no warmup
- AMP: autocast + GradScaler
- Grad clipping: 1.0
- Metric: SMAPE on price scale (expm1 of predictions)
- Checkpointing:
  - Resume from `./checkpoints_deberta-v3-large_clip-large/latest_checkpoint.pth` if present.
  - Save best as `best_model_epoch_{epoch}_{smape:.2f}.pth`
  - Save rolling `latest_checkpoint.pth` each epoch.

Note: The notebook defines `train_model` twice; the second (with gradient accumulation + resume-safe scheduler steps) overrides the first.

## Inference
- Prepare test using the fitted OneHotEncoder and StandardScaler.
- Recreate tokenizer, CLIPProcessor, encoders, and model with the same numeric feature size.
- Load the best checkpoint.
- Predict in batches (often 2× train batch size), expm1 to price, clip to [0, ∞).
- Save submission CSV with columns: sample_id, price.

Filename consistency:
- The loader points to `best_model_45.55.pth` while training saves `best_model_epoch_*`. Align this to the actual best filename.
- The saved submission is `submission_48.39_model.csv` but the printed message names `submission_dl_model.csv`. Make these consistent.

## Results
- Best validation SMAPE: 44.06%
  - Referenced by the checkpoint name used for inference in the notebook (`best_model_44.06.pth`).
  
