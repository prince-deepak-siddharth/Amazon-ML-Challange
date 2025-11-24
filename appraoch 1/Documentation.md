# **ML Challenge 2025: Smart Product Pricing Solution Template**

**Team Name:** The\_Predictors

**Team Members:** Prince Deepak Siddharth

     Vishal Painjane

                 Samhita Mandal

     Yash Patil

 **Submission Date:** October 12, 2025

---

## **1\. Executive Summary**

This solution presents a robust baseline model for predicting product prices by leveraging GPU-accelerated feature engineering on textual and structured data. Our approach systematically deconstructs the provided `catalog_content` and trains a powerful XGBoost model, achieving a validation SMAPE score of 55.60% and establishing a strong initial performance benchmark.

---

## **2\. Methodology Overview**

### **2.1 Problem Analysis**

The core of the challenge is predicting a continuous variable (`price`) from a mix of unstructured text, categorical data, and numerical data. Our initial Exploratory Data Analysis (EDA) revealed two key insights that guided our strategy.

**Key Observations:**

* The `price` distribution is heavily right-skewed, with many low-priced items and a long tail of expensive ones. To handle this and align with the SMAPE evaluation metric, we chose to predict the logarithm of the price (`log(price + 1)`), which resulted in a more normal distribution.  
* The `catalog_content` field is not just a random block of text but contains structured, high-value information like `Item Name`, `Value`, `Unit`, and `Item Pack Quantity (IPQ)`. We created a parsing function to extract these as distinct, high-quality features.

### **2.2 Solution Strategy**

Our strategy was to build a fast and effective baseline using a single, powerful model. We prioritized efficient, GPU-accelerated data preprocessing to handle the large dataset, transforming all available text and tabular data into a unified numerical format suitable for a gradient boosting model.

**Approach Type:** Single Model (XGBoost) **Core Innovation:** The main contribution of this baseline is a robust feature engineering pipeline that parses the structured `catalog_content` and utilizes the RAPIDS ecosystem (`cuDF`, `cuML`) for a high-speed, end-to-end GPU-accelerated workflow.

---

## **3\. Model Architecture**

### **3.1 Architecture Overview**

Our architecture is a classic machine learning pipeline for tabular and text data. Raw data is cleaned and parsed, then converted into a single, wide feature matrix. This matrix is then fed into a Gradient Boosted Decision Tree model (XGBoost) to perform the final price regression.

**Pipeline Flow:** Raw Data (`train.csv`) ⟶ Text & Feature Parsing (`ipq`, `value`, `unit`, `clean_text`) ⟶ GPU Feature Conversion (TF-IDF, One-Hot Encoding) ⟶ Combined Feature Matrix ⟶ XGBoost Model ⟶ Price Prediction

### **3.2 Model Components**

**Text & Tabular Processing Pipeline:**

* **Preprocessing steps:**  
  * Extracted `Item Name`, `Bullet Points`, `Value`, and `Unit` from the `catalog_content` string using regular expressions.  
  * Extracted `Item Pack Quantity (IPQ)` as a separate numerical feature.  
  * Created a `clean_text` feature by combining `Item Name` and `Bullet Points`.  
  * Filled missing `value` entries with the median value of the training set.  
  * Log-transformed the `price` target variable using `np.log1p`.  
* **Model type:** XGBoost Regressor (`xgboost.XGBRegressor`).  
* **Key parameters:**  
  * `tree_method='gpu_hist'` to enable GPU training.  
  * `early_stopping_rounds=100` to prevent overfitting.

**Image Processing Pipeline:**

* *Not implemented in this baseline version.* The `image_path` was generated but not used for feature extraction.

---

## **4\. Model Performance**

### **4.1 Validation Results**

The model was trained on 80% of the training data (60,000 samples) and evaluated on the remaining 20% (15,000 samples).

* **SMAPE Score:** **55.60%**  
* **Other Metrics:** Best Validation RMSE on log-transformed price was **0.71784**.

## **5\. Conclusion**

This baseline approach successfully established a benchmark SMAPE of 55.60% using an XGBoost model on text and tabular data. This confirms the validity of our GPU-accelerated feature engineering pipeline and sets a strong foundation for future improvements, primarily by incorporating the image features.

