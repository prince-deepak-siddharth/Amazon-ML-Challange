# Amazon ML Challenge 2025 — Smart Product Pricing Engine

### Team: **The_Predictors**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Key Insights & Observations](#2-key-insights--observations)
3. [Solution Overview](#3-solution-overview)
4. [End-to-End Workflow](#4-end-to-end-workflow)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Ensemble Strategy](#9-ensemble-strategy)
10. [Evolution of Approaches](#10-evolution-of-approaches)
11. [Results](#11-results)
12. [Tech Stack](#12-tech-stack)
13. [Repository Structure](#13-repository-structure)
14. [How to Reproduce](#14-how-to-reproduce)

---

## 1. Problem Statement

Predict the **price** of a product given multi-modal inputs:
- **Text** — product catalog content (item name, bullet points, metadata)
- **Images** — product photos
- **Structured data** — value, unit, pack quantity

**Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)

$$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

---

## 2. Key Insights & Observations

| Insight | Impact |
|---------|--------|
| SMAPE penalizes relative errors heavily on low-price items | Predict `log(price + 1)` instead of raw price |
| `catalog_content` contains structured metadata hiding in free text | Regex-based extraction of `item_name`, `bullet_points`, `value`, `unit` |
| **IPQ** (Item Pack Quantity) is a strong numerical signal | Dedicated regex parser for pack sizes |
| Some images are missing or corrupted | Fallback to blank white 224×224 placeholder |
| Price distribution is heavily right-skewed | Log transformation normalizes the target |
| Larger transformer encoders consistently yield lower SMAPE | Final model uses DeBERTa-v3-Large + CLIP ViT-L/14 |

---

## 3. Solution Overview

Our final solution is a **weighted ensemble of 3 multi-modal transformer models**, each fusing text, image, and numerical features through a shared MLP regressor head.

```mermaid
graph LR
    A["Model A\nDistilBERT + ViT\nSMAPE: 48.23%"] -->|"w = 0.15"| E
    B["Model B\nRoBERTa-Large + CLIP-Large\nSMAPE: 45.55%"] -->|"w = 0.20"| E
    C["Model C\nDeBERTa-Large + CLIP-Large\nSMAPE: 44.06%"] -->|"w = 0.65"| E
    E["Weighted\nAverage"] --> F["Final Price\nSMAPE: 43.60%"]

    style A fill:#fce4ec,stroke:#c62828,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e8f5e9,stroke:#2e7d32,color:#000
    style E fill:#fff3e0,stroke:#e65100,color:#000
    style F fill:#f3e5f5,stroke:#6a1b9a,color:#000
```

---

## 4. End-to-End Workflow

```mermaid
flowchart TD
    subgraph INPUT ["Raw Data"]
        A1["train.csv\n75,000 samples"]
        A2["test.csv\n75,000 samples"]
        A3["Product Images\nimages_train / images_test"]
    end

    subgraph PARSE ["Feature Parsing & Extraction"]
        B1["parse_catalog_content()\nRegex extraction"]
        B2["extract_ipq()\nPack quantity parser"]
        B3["item_name + bullet_points\n→ clean_text"]
    end

    subgraph PREP ["Data Preparation"]
        C1["Train/Val Split\n90/10, seed=42"]
        C2["OneHotEncoder\nunit field"]
        C3["StandardScaler\nvalue, ipq"]
        C4["Median Imputation\nmissing values"]
    end

    subgraph DATASET ["ProductDataset"]
        D1["Tokenized Text\ninput_ids + attention_mask"]
        D2["Processed Image\npixel_values [3,224,224]"]
        D3["Numeric Tensor\nscaled value + ipq + unit OHE"]
    end

    subgraph MODEL ["MultiModalRegressor"]
        E1["Text Branch\nDeBERTa → 512d"]
        E2["Image Branch\nCLIP ViT → 512d"]
        E3["Numeric Branch\nLinear → 128d"]
        E4["Concatenate\n1152d"]
        E5["MLP Head\n1152→512→ReLU→Dropout→1"]
    end

    subgraph TRAIN ["Training Loop"]
        F1["SmoothL1Loss on log_price"]
        F2["AdamW lr=1e-5"]
        F3["AMP Mixed Precision"]
        F4["Gradient Accumulation ×2"]
        F5["Checkpoint Resume"]
    end

    subgraph INFER ["Inference"]
        G1["expm1 → price scale"]
        G2["clip negatives → 0"]
    end

    subgraph ENSEMBLE ["Weighted Ensemble"]
        H1["Model A × 0.15"]
        H2["Model B × 0.20"]
        H3["Model C × 0.65"]
        H4["Grid-search\noptimal weights"]
    end

    I["submission.csv\nsample_id, price"]

    INPUT --> PARSE
    PARSE --> PREP
    PREP --> DATASET
    DATASET --> MODEL
    D1 --> E1
    D2 --> E2
    D3 --> E3
    E1 --> E4
    E2 --> E4
    E3 --> E4
    E4 --> E5
    MODEL --> TRAIN
    TRAIN --> INFER
    INFER --> ENSEMBLE
    H1 --> H4
    H2 --> H4
    H3 --> H4
    ENSEMBLE --> I

    style INPUT fill:#e3f2fd,stroke:#1565c0,color:#000
    style PARSE fill:#fff3e0,stroke:#e65100,color:#000
    style PREP fill:#f3e5f5,stroke:#6a1b9a,color:#000
    style DATASET fill:#e8f5e9,stroke:#2e7d32,color:#000
    style MODEL fill:#fce4ec,stroke:#c62828,color:#000
    style TRAIN fill:#fffde7,stroke:#f57f17,color:#000
    style INFER fill:#e0f7fa,stroke:#00695c,color:#000
    style ENSEMBLE fill:#fbe9e7,stroke:#bf360c,color:#000
    style I fill:#c8e6c9,stroke:#1b5e20,color:#000
```

---

## 5. Data Processing Pipeline

### 5.1 Text Feature Extraction

```mermaid
flowchart TD
    A["catalog_content\nraw string"] --> B["parse_catalog_content()"]
    A --> C["extract_ipq()"]

    B --> B1["item_name\nregex: Item Name:...Bullet Point"]
    B --> B2["bullet_points\nregex: Bullet Point \d+:..."]
    B --> B3["value\nregex: Value: [\d.]+"]
    B --> B4["unit\nregex: Unit: \w+"]

    C --> C1["IPQ\npack of X · X-pack\nX count · X ct · case of X\ndefault: 1"]

    B1 --> D["clean_text\n= item_name + bullet_points"]
    B2 --> D

    style A fill:#fff3e0,stroke:#e65100,color:#000
    style B fill:#e3f2fd,stroke:#1565c0,color:#000
    style C fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e8f5e9,stroke:#2e7d32,color:#000
    style C1 fill:#f3e5f5,stroke:#6a1b9a,color:#000
```

### 5.2 Numerical Feature Engineering

| Step | Operation |
|------|-----------|
| 1 | Extract `value` (float), `ipq` (int), `unit` (string) |
| 2 | Fill missing `value` with training set **median** |
| 3 | **One-Hot Encode** `unit` via `sklearn.OneHotEncoder` |
| 4 | **StandardScaler** on `[value, ipq]` (fit on train, transform val/test) |
| 5 | Final numeric tensor = `[scaled_value, scaled_ipq, unit_one_hot...]` |

### 5.3 Image Processing

```mermaid
flowchart TD
    A["image_link"] --> B["Extract filename\nstr.split('/').last"]
    B --> C{"Load from disk"}
    C -->|"Success"| D["PIL Image → RGB"]
    C -->|"Missing/Corrupt"| E["Blank white\n224×224 RGB placeholder"]
    D --> F["CLIPProcessor\nAutoImageProcessor"]
    E --> F
    F --> G["Resize"]
    G --> H["Center Crop"]
    H --> I["Normalize"]
    I --> J["pixel_values\ntensor [3, 224, 224]"]

    style A fill:#fff3e0,stroke:#e65100,color:#000
    style C fill:#fffde7,stroke:#f57f17,color:#000
    style D fill:#e8f5e9,stroke:#2e7d32,color:#000
    style E fill:#fce4ec,stroke:#c62828,color:#000
    style F fill:#e3f2fd,stroke:#1565c0,color:#000
    style J fill:#c8e6c9,stroke:#1b5e20,color:#000
```

### 5.4 Text Tokenization

- **Tokenizer:** `AutoTokenizer` matching the text encoder model
- **Max Length:** 128 tokens
- **Strategy:** truncation + padding to `max_length`
- **Output:** `input_ids` + `attention_mask`

### 5.5 Target Transformation

```python
log_price = np.log1p(price)       # Training target
predicted_price = np.expm1(pred)   # Back-transform at inference
```

---

## 6. Model Architecture

### 6.1 MultiModalRegressor (Final Model — DeBERTa + CLIP)

```mermaid
flowchart TD
    subgraph TEXT ["Text Branch"]
        T1["input_ids\nattention_mask"] --> T2["DeBERTa-v3-Large\n435M params"]
        T2 --> T3["last_hidden_state"]
        T3 --> T4["Mean Pooling"]
        T4 --> T5["Linear\n1024 → 512"]
    end

    subgraph IMAGE ["Image Branch"]
        I1["pixel_values"] --> I2["CLIP ViT-L/14\n427M params"]
        I2 --> I3["pooler_output"]
        I3 --> I5["Linear\n1024 → 512"]
    end

    subgraph NUMERIC ["Numeric Branch"]
        N1["value + ipq\n+ unit_one_hot"] --> N5["Linear\nN → 128"]
    end

    T5 --> CAT["Concatenate\n512 + 512 + 128 = 1152d"]
    I5 --> CAT
    N5 --> CAT

    subgraph MLP ["MLP Head"]
        M1["Linear 1152 → 512"]
        M2["ReLU"]
        M3["Dropout 0.3"]
        M4["Linear 512 → 1"]
        M1 --> M2 --> M3 --> M4
    end

    CAT --> M1
    M4 --> OUT["log_price"]

    style TEXT fill:#e3f2fd,stroke:#1565c0,color:#000
    style IMAGE fill:#fff3e0,stroke:#e65100,color:#000
    style NUMERIC fill:#f3e5f5,stroke:#6a1b9a,color:#000
    style MLP fill:#fce4ec,stroke:#c62828,color:#000
    style CAT fill:#fffde7,stroke:#f57f17,color:#000
    style OUT fill:#c8e6c9,stroke:#1b5e20,color:#000
```

### 6.2 Model Variants

| Component | Model A | Model B | Model C (Final) |
|-----------|---------|---------|-----------------|
| **Text Encoder** | DistilBERT-base-uncased | RoBERTa-Large | DeBERTa-v3-Large |
| **Text Params** | ~66M | ~355M | ~435M |
| **Image Encoder** | ViT-base-patch16-224 | CLIP ViT-L/14 | CLIP ViT-L/14 |
| **Image Params** | ~86M | ~427M | ~427M |
| **Total Params** | ~152M | ~782M | **738.83M** |
| **Pooling** | pooler_output | pooler_output | mean-pool (last_hidden_state) |
| **Val SMAPE** | 48.23% | 45.55% | **44.06%** |

---

## 7. Training Pipeline

### 7.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| Loss Function | SmoothL1Loss (Huber Loss) |
| Scheduler | Linear (no warmup) |
| Physical Batch Size | 16 |
| Gradient Accumulation Steps | 2 |
| **Effective Batch Size** | **32** |
| Epochs | 30 |
| Max Sequence Length | 128 |
| Dropout | 0.3 |
| Gradient Clipping | 1.0 |
| Mixed Precision | AMP (autocast + GradScaler) |
| Train/Val Split | 90/10 (seed=42) |

### 7.2 Training Flow

```mermaid
flowchart TD
    START(["Start Training"]) --> RESUME{"Checkpoint\nexists?"}
    RESUME -->|"Yes"| LOAD["Load latest_checkpoint.pth\nmodel + optimizer + scheduler\n+ scaler + epoch + best_smape"]
    RESUME -->|"No"| INIT["Initialize from scratch\nepoch = 0"]
    LOAD --> EPOCH
    INIT --> EPOCH

    EPOCH["for epoch in range start_epoch to 30"] --> TRAIN_PHASE

    subgraph TRAIN_PHASE ["Training Phase"]
        TP1["Batch from train_loader"] --> TP2["Forward pass\nautocast FP16"]
        TP2 --> TP3["loss = SmoothL1Loss\npred vs log_price"]
        TP3 --> TP4["loss /= GRAD_ACCUM_STEPS"]
        TP4 --> TP5["Backward pass\naccumulate gradients"]
        TP5 --> TP6{"i+1 %\nGRAD_ACCUM\n== 0 ?"}
        TP6 -->|"Yes"| TP7["Unscale gradients\nClip max_norm=1.0\nOptimizer step\nScheduler step\nZero gradients"]
        TP6 -->|"No"| TP1
        TP7 --> TP8{"More\nbatches?"}
        TP8 -->|"Yes"| TP1
        TP8 -->|"No"| VAL_PHASE
    end

    subgraph VAL_PHASE ["Validation Phase"]
        VP1["Batch from val_loader"] --> VP2["Forward pass\nno_grad + autocast"]
        VP2 --> VP3["pred_price = expm1 pred_log"]
        VP3 --> VP4["Compute SMAPE\non price scale"]
        VP4 --> VP5{"More\nbatches?"}
        VP5 -->|"Yes"| VP1
        VP5 -->|"No"| CKPT
    end

    subgraph CKPT ["Checkpointing"]
        CK1{"val_smape\n< best_smape?"}
        CK1 -->|"Yes"| CK2["Save best_model_epoch_N_XX.XX.pth"]
        CK1 -->|"No"| CK3
        CK2 --> CK3["Save latest_checkpoint.pth\nFull state: model + optimizer\n+ scheduler + scaler + epoch"]
    end

    CKPT --> NEXT{"More\nepochs?"}
    NEXT -->|"Yes"| EPOCH
    NEXT -->|"No"| DONE(["Training Complete\nBest SMAPE: 44.06%"])

    style TRAIN_PHASE fill:#e3f2fd,stroke:#1565c0,color:#000
    style VAL_PHASE fill:#e8f5e9,stroke:#2e7d32,color:#000
    style CKPT fill:#fff3e0,stroke:#e65100,color:#000
    style DONE fill:#c8e6c9,stroke:#1b5e20,color:#000
```

### 7.3 Resume-Safe Checkpointing

The training pipeline supports **full checkpoint resumption**, saving:
- Model weights
- Optimizer state (momentum buffers, etc.)
- LR Scheduler state
- AMP GradScaler state
- Current epoch & best SMAPE

This allowed resuming after epoch 20 with reduced batch size (32 → 16) and gradient accumulation (×2) to avoid OOM errors while maintaining the same effective batch size.

---

## 8. Inference Pipeline

```mermaid
flowchart TD
    A["Test Data\ntest.csv + test images"] --> B["parse_catalog_content\nextract_ipq"]
    B --> C["prepare_inference_data"]
    C --> C1["encoder.transform\nfitted OneHotEncoder"]
    C --> C2["scaler.transform\nfitted StandardScaler"]
    C1 --> D
    C2 --> D
    D["ProductDataset\nis_train=False"] --> E["Load best checkpoint\nbest_model_epoch_29_44.06.pth"]
    E --> F["Batch prediction\n2x train batch size"]

    subgraph POST ["Post-Processing"]
        G["autocast FP16"] --> H["model → log_price preds"]
        H --> I["np.expm1 → price scale"]
        I --> J["clip lower=0\nnon-negative prices"]
    end

    F --> G
    J --> K["submission.csv\nsample_id, price"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style D fill:#e8f5e9,stroke:#2e7d32,color:#000
    style POST fill:#fff3e0,stroke:#e65100,color:#000
    style K fill:#c8e6c9,stroke:#1b5e20,color:#000
```

---

## 9. Ensemble Strategy

### 9.1 Weight Optimization via Grid Search

Optimal weights were found by **grid search** over the validation set:

```mermaid
flowchart TD
    A["Validation Predictions\nfrom 3 models"] --> B["Grid Search"]

    subgraph GRID ["Weight Grid Search"]
        B --> C["w_a ∈ 0.00, 0.05 ... 0.20"]
        C --> D["w_b ∈ 0.00, 0.05 ... 0.70"]
        D --> E["w_c = 1.0 − w_a − w_b"]
        E --> F{"w_c ≥ 0?"}
        F -->|"Yes"| G["ensemble = w_a·pred_A\n+ w_b·pred_B + w_c·pred_C"]
        F -->|"No"| C
        G --> H["Compute SMAPE\nvs y_true"]
        H --> I{"SMAPE <\nbest_smape?"}
        I -->|"Yes"| J["Update best_weights\nUpdate best_smape"]
        I -->|"No"| C
        J --> C
    end

    GRID --> K["Optimal Weights\nA=0.15, B=0.20, C=0.65\nSMAPE: 43.5961%"]

    style GRID fill:#e3f2fd,stroke:#1565c0,color:#000
    style K fill:#c8e6c9,stroke:#1b5e20,color:#000
```

### 9.2 Final Weights

| Model | Architecture | Individual SMAPE | Ensemble Weight |
|-------|-------------|------------------|-----------------|
| **A** | DistilBERT + ViT | 48.23% | 0.15 |
| **B** | RoBERTa-Large + CLIP-Large | 45.55% | 0.20 |
| **C** | DeBERTa-Large + CLIP-Large | 44.06% | **0.65** |

### 9.3 Final Ensembled SMAPE: **43.5961%**

The ensemble outperforms the best single model (44.06%) by ~0.46 percentage points.

---

## 10. Evolution of Approaches

```mermaid
flowchart LR
    A["Approach 1\nXGBoost\nTabular Only\n\nSMAPE: 55.60%"] --> B["Approach 2\nDistilBERT + ViT\nMulti-Modal\n\nSMAPE: 48.23%"]
    B --> C["Approach 3\nRoBERTa-Large\n+ CLIP-Large\n\nSMAPE: 45.55%"]
    C --> D["Approach 4\nDeBERTa-Large\n+ CLIP-Large\n\nSMAPE: 44.06%"]
    D --> E["Final\nWeighted Ensemble\nAll 3 Models\n\nSMAPE: 43.60%"]

    B -.->|"w=0.15"| E
    C -.->|"w=0.20"| E
    D -.->|"w=0.65"| E

    style A fill:#ffcdd2,stroke:#c62828,color:#000
    style B fill:#fff9c4,stroke:#f57f17,color:#000
    style C fill:#ffe0b2,stroke:#e65100,color:#000
    style D fill:#c8e6c9,stroke:#2e7d32,color:#000
    style E fill:#b2dfdb,stroke:#00695c,color:#000
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#e3f2fd'}}}%%
xychart-beta
    title "SMAPE Improvement Across Approaches"
    x-axis ["XGBoost", "DistilBERT+ViT", "RoBERTa+CLIP", "DeBERTa+CLIP", "Ensemble"]
    y-axis "SMAPE %" 40 --> 58
    bar [55.60, 48.23, 45.55, 44.06, 43.60]
```

| Approach | Model | Key Change | SMAPE |
|----------|-------|------------|-------|
| 1 | XGBoost (GPU) | Baseline — TF-IDF + tabular features, no images | 55.60% |
| 2 | DistilBERT + ViT | First multi-modal (text+image+numeric) deep learning model | 48.23% |
| 3 | RoBERTa-Large + CLIP-Large | Scaled up text & image encoders | 45.55% |
| 4 | DeBERTa-v3-Large + CLIP-Large | Strongest encoders, mean-pooling, gradient accumulation, 30 epochs | 44.06% |
| **Final** | **Weighted Ensemble (A+B+C)** | **Grid-searched weights across 3 models** | **43.60%** |

---

## 11. Results

### 11.1 Validation Performance (10% Holdout)

| Metric | Best Single Model | Ensemble |
|--------|-------------------|----------|
| **SMAPE** | 44.06% | **43.5961%** |

### 11.2 Model Parameter Counts

| Model | Total Parameters |
|-------|-----------------|
| Model A (DistilBERT + ViT) | ~152M |
| Model B (RoBERTa-Large + CLIP-Large) | ~782M |
| Model C (DeBERTa-Large + CLIP-Large) | **738.83M** |

### 11.3 Dataset Statistics

| Split | Samples |
|-------|---------|
| Training | 75,000 (4 columns) |
| Test | 75,000 (3 columns) |
| Train split | 67,500 (90%) |
| Validation split | 7,500 (10%) |

---

## 12. Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Deep Learning** | PyTorch, PyTorch AMP |
| **Transformers** | HuggingFace Transformers (AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor) |
| **NLP Models** | DeBERTa-v3-Large, RoBERTa-Large, DistilBERT |
| **Vision Models** | CLIP ViT-L/14, ViT-base-patch16-224 |
| **ML Baseline** | XGBoost (GPU — `gpu_hist`), RAPIDS (cuDF, cuML) |
| **Preprocessing** | scikit-learn (OneHotEncoder, StandardScaler, train_test_split) |
| **Data** | Pandas, NumPy |
| **Image** | Pillow (PIL) |
| **Compute** | NVIDIA GPU (CUDA) |
| **Environment** | Jupyter Notebook |

---

## 13. Repository Structure

```
Amazon-ML-Challange/
│
├── README.md                              # Project overview (root)
│
├── dataset/
│   ├── train.csv                          # 75,000 training samples
│   └── test.csv                           # 75,000 test samples
│
├── images_train/                          # Training product images
├── images_test/                           # Test product images
│
├── appraoch 1/                            # XGBoost baseline
│   ├── Documentation.md
│   ├── test.ipynb
│   └── submissionxgb.csv
│
├── approach 2/                            # DistilBERT + ViT
│   ├── test1 (1).ipynb
│   ├── submission_45.55_model.csv
│   └── submission_48.39_model.csv
│
├── approach 3/                            # RoBERTa-Large + CLIP-Large
│   ├── approach.md
│   ├── test_roberta-large_clip-large.ipynb
│   ├── submission_45.55_model.csv
│   ├── submission_average_ensemble.csv
│   └── submission_average_ensemble1.csv
│
├── approach 4/                            # DeBERTa-v3-Large + CLIP-Large
│   ├── approach4.md
│   ├── main.ipynb
│   └── submission_deberta_large_clip-large_44.06.csv
│
└── Final_submission/                      # FINAL SOLUTION
    ├── README.md                          # This file
    ├── main7.ipynb                        # Complete pipeline notebook
    └── submission_ensembled_final2.csv    # Final submission file
```

---

## 14. How to Reproduce

### Prerequisites

```bash
pip install torch torchvision transformers scikit-learn pandas numpy pillow tqdm timm
```

### Steps

1. **Place data** in `../dataset/` (`train.csv`, `test.csv`) and images in `../images_train/`, `../images_test/`

2. **Train individual models** — Run the training cells in `main7.ipynb`:
   - Configure `Config.TEXT_MODEL` and `Config.IMAGE_MODEL` for each variant
   - Training supports resume from checkpoint

3. **Generate validation predictions** — Save per-model validation predictions for weight optimization

4. **Find ensemble weights** — Run the grid search cell to find optimal blending weights on validation set

5. **Generate final submission** — Apply ensemble weights to test predictions → `submission_ensembled_final2.csv`

### GPU Requirements

| Model | Approx. VRAM |
|-------|-------------|
| Model A (DistilBERT + ViT) | ~8 GB |
| Model B (RoBERTa + CLIP-L) | ~20 GB |
| Model C (DeBERTa + CLIP-L) | ~20 GB |

> Mixed precision (AMP) and gradient accumulation are used to fit large models in limited VRAM.

---

<p align="center">
  <b>Final SMAPE: 43.5961%</b><br>
  <i>Amazon ML Challenge 2025 — Team The_Predictors</i>
</p>
