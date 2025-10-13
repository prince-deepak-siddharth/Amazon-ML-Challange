# Approach

This document explains the end-to-end approach used in `test_roberta-large_clip-large.ipynb` to predict product prices by fusing text, image, and structured features.

## Objective

Predict price from:
- Text: item_name + bullet points (RoBERTa-Large)
- Image: product image (CLIP ViT-L/14)
- Numeric: parsed value and IPQ, plus one-hot unit

Target is modeled on log scale (log1p(price)).

## Data and preprocessing

- Input CSVs: `../dataset/train.csv`, `../dataset/test.csv`
- Images: `../images_train/`, `../images_test/` derived from `image_link`

Feature parsing from `catalog_content`:
- parse_catalog_content(text)
  - item_name
  - bullet_points (concatenated)
  - value (float; NaN if missing)
  - unit (string; default "unknown")
- extract_ipq(text)
  - Parses pack size from patterns like “pack of X”, “X-pack”, “X count/ct”, “case of X”; defaults to 1

Modeling text:
- clean_text = item_name + " " + bullet_points

Cleaning and engineering:
- Drop: catalog_content, filename, item_name, bullet_points, image_link
- Impute value with train median; apply to test
- One-hot encode unit with OneHotEncoder(handle_unknown="ignore")
- Scale numeric ["value", "ipq"] with StandardScaler (fit on train, apply to val/test)
- Train/val split: 90/10 with seed=42

## Tokenization and image processing

- Text: AutoTokenizer for `roberta-large`, max_length=128, truncation/padding
- Images: CLIPProcessor for `openai/clip-vit-large-patch14` (resize, center-crop, normalize)
- Missing/corrupt images replaced with a blank RGB image

## Dataset

ProductDataset returns:
- input_ids, attention_mask (text)
- pixel_values (image)
- numeric tensor: scaled [value, ipq] + unit_* one-hots
- For training: also returns log_price and price

## Model

MultiModalRegressor with three branches:
- Text encoder: AutoModel(roberta-large), uses pooler_output (hidden_size=1024) → Linear to 512
- Image encoder: CLIPModel(...).vision_model, uses pooler_output (hidden_size=1024) → Linear to 512
- Numeric head: Linear(num_numeric_features → 128)
- Fusion: concat [512, 512, 128] → Head: Linear(1152→512) → ReLU → Dropout(0.3) → Linear(512→1)
- Output predicts log_price

## Training

- Device: CUDA if available
- Batch size: 32
- Epochs: 10
- Optimizer: AdamW(lr=1e-5)
- Loss: SmoothL1Loss on log_price
- Scheduler: linear with no warmup
- AMP mixed precision (autocast + GradScaler)
- Grad clipping: 1.0
- Metric: SMAPE on price scale (expm1 of predictions)
- Checkpoint: save best by lowest validation SMAPE to `./checkpoints_roberta-large_clip-large14_models/best_model_XX.XX.pth`

## Inference

- Transform test with the fitted OneHotEncoder and StandardScaler
- Recreate tokenizer, CLIPProcessor, and encoders
- Build the same MultiModalRegressor with matching numeric_feature_size
- Load best checkpoint
- Predict in batches (often double batch size for inference)
- Convert predictions via expm1, clip negatives to 0
- Save submission as CSV with columns: sample_id, price


## Results

- Best validation SMAPE (10% holdout): 45.55%
  - Checkpoint saved as: `best_model_45.55.pth`
- Submission artifact: `submission_45.55_model.csv`

