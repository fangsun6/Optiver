# Optiver Trading at the Close - LightGBM Model Pipeline

This repository contains a complete modeling pipeline for the [Optiver Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) Kaggle competition. The goal is to predict future price movements of stocks using high-frequency market features. The solution includes advanced feature engineering, a custom purged time series cross-validation scheme, and training a LightGBM model with GPU acceleration.

---

## 📁 Project Structure

### Key Components

- **Data Loading**
  - Reads `train.csv` and prepares target features (e.g., `target_shift1`)

- **Feature Engineering**
  - Computes:
    - Price and size-based imbalance features
    - Spread, momentum, pressure indicators
    - Statistical aggregations and rolling window stats using `Polars`
    - Triplet imbalance features using `Numba`

- **Memory Optimization**
  - `reduce_mem_usage(df)` downcasts numeric columns to optimize memory usage

- **Cross-Validation**
  - `PurgedGroupTimeSeriesSplit`: custom time-aware CV that prevents leakage between folds using group gaps

- **Model Training**
  - Uses LightGBM with the following key settings:
    - GPU acceleration
    - MAE loss
    - Early stopping
  - Trains on 5 folds and saves each model to disk

- **Real-Time Inference**
  - Integrates with `optiver2023` environment
  - Online prediction using ensemble averaging of all trained models

---

## 🚀 Features

- ✅ Over 100+ engineered features
- ✅ GPU-accelerated LightGBM training
- ✅ Custom purged time-series cross-validation
- ✅ Real-time prediction loop with feature caching
- ✅ Optimized for low memory usage and high speed

---

## 🧰 Requirements

```bash
pip install numpy pandas lightgbm scikit-learn numba polars joblib

