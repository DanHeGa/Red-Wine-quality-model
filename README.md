# 🍷 Red Wine Quality Model
Machine Learning project based on the **Red Wine Quality** dataset from Kaggle.  
The goal is to train regression models and track experiments using **MLflow** and **DagsHub**.

---

## 📌 Project Overview

This project aims to:
- Predict wine quality using regression models
- Track experiments with **MLflow**
- Visualize experiment results remotely on **DagsHub**
- Use **Conda** for environment management

---

## 🗂️ Dataset

- **Source:** [Kaggle – Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)  
- **Target variable:** `quality` (wine quality score)

---

## 🛠️ Environment Setup

This project uses **Conda** as the environment manager.  

If you don’t have Conda installed, refer to the official documentation:  
👉 [Conda Documentation](https://docs.conda.io/en/latest/)

### 1️⃣ Create the environment

```bash
conda env create -f environment.yml
conda activate wine_model_env
```

## Running the project 
### With default parameters
```bash
python demo.py
```
### With custom parameters
```bash
python demo.py 0.5 0.5
```

