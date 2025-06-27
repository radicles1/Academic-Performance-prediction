# Academic-Performance-prediction
# Enhanced Identification of Behavioral Analysis and Academic Performance using Optimized Binarized Simplicial Convolutional Network for Adolescents

## Overview

This repository provides an end-to-end pipeline for data preprocessing, feature extraction, behavioral and academic performance analysis, and prediction using an Multi Component Attention Graph Convolutional Neural Network (BSCNN) tailored for adolescent student data. The goal is to identify and analyze key educational and behavioral metrics (IQ, CQ, EQ, SQ, AQ) and predict these soft labels for individual students.

## Project Structure

```
├── data/
│   ├── Student__Dataset.csv         # Raw student dataset
│   └── Students_Grading_Dataset.csv # Backup grading dataset
│
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook with preprocessing, feature extraction, analysis, and modeling code
│
├── src/
│   ├── preprocessing.py             # Data cleaning and FRIF filtering
│   ├── feature_extraction.py        # STET transform and statistical feature extraction
│   ├── encoding_pipeline.py         # Single-record encoding function
│   └── model.py                     # Binarized Simplicial CNN implementation
│
├── outputs/
│   ├── STET_Extracted_Features.csv  # Extracted features CSV
│   └── remya.keras                  # Trained model weights
│
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## Features

* **Data Cleaning & Smoothing**: Handles missing values, applies Fast Resampled Iterative Filtering (FRIF)-style smoothing, and normalizes numerical features.
* **Feature Extraction**: Applies Synchro-Transient Extracting Transform (STET) using Hilbert transform, extracting statistical measures (mean, std, skewness, kurtosis, entropy) for each signal.
* **Academic & Behavioral Metrics**:

  * Intelligence Quotient (IQ)
  * Cognitive Quotient (CQ)
  * Emotional Quotient (EQ)
  * Social Quotient (SQ)
  * Adversity Quotient (AQ)
* **Encoding Pipeline**: Pre-fitted scaler pipelines for both academic and behavioral features; supports single-record processing for inference.
* **Model**: Custom Binarized Simplicial Convolutional Neural Network built with TensorFlow/Keras; predicts multi-output soft labels.
* **Visualization**: Comprehensive plots for distribution, correlation, and per-student prediction results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/optimized-bscnn-adolescents.git
   cd optimized-bscnn-adolescents
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**

   * Place raw CSV files under the `data/` directory.
   * Ensure column names match expected schema (e.g., `Midterm_Score`, `Attendance (%)`, `Stress_Level (1-10)`, etc.).

2. **Run Notebook**

   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

   Follow the notebook cells to:

   * Clean and smooth data
   * Extract STET features
   * Compute IQ, CQ, EQ, SQ, AQ metrics
   * Train and evaluate the BSCNN model
   * Generate visualizations


## Dependencies

See `requirements.txt` for full list. Key packages include:

* pandas
* numpy
* scikit-learn
* scipy
* matplotlib
* seaborn
* tensorflow

## Results

* Achieved validation metrics:

  * MSE, RMSE, MAE, MAPE, R² scores for overall and per-output labels.
