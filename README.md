# Amazon ML Challenge 2025 — Smart Product Pricing  
### Team: The_Predictors  
**Members:** Prince Deepak Siddharth, Vishal Painjane, Samhita Mandal, Yash Patil  

---

## 1. Introduction  
This repository contains our complete solution for the **Amazon ML Challenge 2025 – Smart Product Pricing**, where the task is to predict product prices using **multi-modal inputs**: text, images, and structured metadata.

We treat this as a **multi-modal regression problem** and train three different transformer-based architectures. A final **weighted ensemble** of these models provides the best performance.

---

## 2. Problem Understanding  
The objective is to estimate the continuous price value of products.  
Since the evaluation metric **SMAPE (Symmetric Mean Absolute Percentage Error)** penalizes relative errors, especially for low-price items, directly predicting raw price is not ideal.

### Key Observations  
- SMAPE is sensitive for small price values → predict **log(price + 1)** instead of price.  
- The semi-structured field `catalog_content` contains rich metadata.  
- Extracted features like **Item Name, Bullet Points, Value, Unit, and IPQ (Item Pack Quantity)** improve accuracy.  
- **IPQ** is often the strongest numerical signal.  
- Some images were missing/corrupted → replaced with blank white placeholder images.

---

## 3. Overall Approach  
Our final solution is a **weighted ensemble** of three multi-modal models that each combine:

- A **text transformer encoder**,  
- A **vision transformer**,  
- A **numerical feature encoder**,  

The encoders output vector embeddings which are projected to uniform dimensions, concatenated, and passed into an **MLP regressor** to predict **log(price)**.

The ensemble improves performance compared to any individual model.

---

## 4. Data Processing

### 4.1 Text Processing  
- Use regex to extract structured metadata from `catalog_content`:  
  - `item_name`  
  - `bullet_points`  
  - `value`  
  - `unit`  
  - `IPQ` (default = 1 if not found)  
- Create `clean_text = item_name + bullet_points`.  
- Tokenize using the corresponding `AutoTokenizer` for each model.  
- Maximum sequence length: **128 tokens**.

### 4.2 Image Processing  
- Extract `sample_id` from image URLs.  
- Replace missing/corrupt images with a white placeholder.  
- Use model-specific processors (`CLIPProcessor` or `AutoImageProcessor`) to apply:  
  - resizing  
  - center cropping  
  - normalization  

### 4.3 Numerical Features  
- One-hot encode the `unit` field.  
- Replace missing `value` with median.  
- Extract `IPQ` as a separate feature.  
- Normalize continuous features.

---

## 5. Model Architecture  

All three models follow this architecture:

Text Input ─► Text Encoder ─┐
Image Input ─► Image Encoder ──► Linear Projection ─► Concatenate ─► MLP ─► log_price
Numeric Input ─► Numerical Encoder ─┘


Each encoder produces embeddings that are projected into a consistent dimension before concatenation.

---

## 6. Models Used

### Model 1: DistilBERT + ViT  
- **Text Encoder:** distilbert-base-uncased (~66M params)  
- **Image Encoder:** google/vit-base-patch16-224-in21k (~86M params)  
- **MLP:** ~152M trainable parameters  
- **Validation SMAPE:** 48.23%

### Model 2: RoBERTa-Large + CLIP-Large  
- **Text Encoder:** roberta-large (~355M params)  
- **Image Encoder:** openai/clip-vit-large-patch14 (~427M params)  
- **MLP:** ~782M trainable parameters  
- **Validation SMAPE:** 45.55%

### Model 3: DeBERTa-Large + CLIP-Large  
- **Text Encoder:** microsoft/deberta-v3-large (~435M params)  
- **Image Encoder:** openai/clip-vit-large-patch14 (~427M params)  
- **MLP:** ~738M trainable parameters  
- **Validation SMAPE:** 44.06%

---

## 7. Final Ensemble Strategy  

We combine predictions from all three models using a weighted average:

| Model | Weight |
|--------|--------|
| DistilBERT + ViT | 0.15 |
| RoBERTa-Large + CLIP-Large | 0.20 |
| DeBERTa-Large + CLIP-Large | 0.65 |

**Final Ensemble Validation SMAPE:** **43.5961%**

The weighted ensemble significantly outperformed all individual models.

---

## 8. Conclusion  
This solution demonstrates that:  
- Multi-modal architectures (text + image + numeric) improve price prediction accuracy.  
- Extracting structured information such as IPQ and unit from catalog text is crucial.  
- Predicting **log(price + 1)** aligns well with SMAPE.  
- A weighted ensemble of diverse transformer models provides substantial performance gains.

Our final model achieved a SMAPE of **43.5961%**, outperforming all standalone models.

---
