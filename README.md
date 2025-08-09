 
# Advanced MLOps Pipeline: Poisoning, Defense & Responsible AI 🛡️

This project demonstrates a full-cycle, production-grade MLOps pipeline built on GCP. It showcases a data poisoning attack, an automated defense mechanism, and a suite of Responsible AI checks for model explainability, fairness, and data drift monitoring.

---

## 🔧 Features Implemented

### ✅ CI/CD Automation with GitHub Actions
- Every PR to `main` triggers a comprehensive workflow that:
  - Pulls the correct data version from GCS using **DVC**.
  - **Validates data quality** to detect label poisoning and fails the build if corruption is found.
  - Runs training, evaluation, and `pytest` unit tests.
  - Generates and uploads a full suite of artifacts, including performance plots and reports.
  - Posts a summary report back to the PR using CML.

### 📊 End-to-End MLOps Tooling
- **DVC:** For versioning datasets (clean, poisoned, biased) to ensure reproducible experiments.
- **MLflow:** A central server on a GCP VM for tracking experiment parameters, metrics, and model artifacts.
- **GCP:** Used for hosting the MLflow server (Compute Engine) and for remote artifact/data storage (Cloud Storage).

### 👾 Data Poisoning & Mitigation (Week 8)
- **Label-Flipping Attack:** `src/poison_data.py` was used to corrupt the dataset at various levels (10%, 25%).
- **Automated Defense:** `src/check_labels.py` was integrated into the CI pipeline as a validation gate to automatically detect and block poisoned data.

### 🧠 Responsible AI & Monitoring (Week 9)
- **Explainability (SHAP):** `src/generate_explanations.py` creates plots to show which features drive model predictions.
- **Fairness Auditing (Fairlearn):** `src/check_fairness.py` assesses the model for bias against a sensitive feature (`location`). The pipeline was tested with both randomly assigned and intentionally biased data.
- **Data Drift Detection (Evidently AI):** `src/check_drift.py` generates an HTML report comparing the statistical properties of different datasets to detect drift.

---

## 📂 Project Structure


iris_pipeline/
├── src/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluates model performance
│   ├── plot_metrics.py       # Generates performance plots
│   ├── poison_data.py        # Attack script: flips labels
│   ├── induce_bias.py        # Creates a biased dataset for fairness tests
│   ├── prepare_data.py       # Adds 'location' column to data
│   ├── check_labels.py       # Defense script: detects label poisoning
│   ├── check_fairness.py     # Runs Fairlearn analysis
│   ├── check_drift.py        # Runs Evidently AI analysis
│   └── generate_explanations.py # Runs SHAP analysis
├── tests/
│   ├── test_model.py         # Unit tests for model accuracy
│   └── test_data_validation.py # Unit tests for data integrity
├── .github/
│   └── workflows/
│       └── main.yml          # GitHub Actions CI/CD workflow
├── data/
│   ├── iris.csv
│   └── iris.csv.dvc          # DVC pointer to the data version
└── requirements.txt


---

## 🔬 Interpreting the Responsible AI Reports

The pipeline generates several reports in the `artifacts/` directory. Here's how to interpret them:

### 1. Explainability (SHAP)
- **File:** `shap_summary_global.png`
- **Purpose:** Shows which features have the biggest impact on the model's predictions overall.
- **Interpretation:** Look for the longest bars. In this project, **`petal_length`** and **`petal_width`** will be the most important features, indicating the model relies heavily on them to classify flowers.

- **Files:** `shap_force_plot_single.html` & `shap_force_plot_all.html`
- **Purpose:** These interactive plots explain individual predictions.
- **Interpretation:**
    - **Red features** push the prediction towards a certain class.
    - **Blue features** push the prediction away from it.
    - This helps you answer "Why did the model predict 'setosa' for this specific flower?"

### 2. Fairness (Fairlearn)
- **File:** `fairness_report.json`
- **Purpose:** Checks if the model is biased towards one group in the `location` column.
- **Interpretation:** Look at the `demographic_parity_difference` for each class.
    - A value **close to 0** means the model is fair; it predicts that class at roughly the same rate for both location 0 and location 1.
    - A value **far from 0** (e.g., > 0.1) indicates potential bias. When you run this on the biased dataset, you will see a significant non-zero value for `virginica`, proving the bias was detected.

### 3. Data Drift (Evidently AI)
- **File:** `data_drift_and_summary_report.html`
- **Purpose:** Creates a detailed visual report comparing the "reference" (clean) dataset to the "current" (potentially changed) dataset.
- **Interpretation:** Open the HTML file in a browser.
    - Look for features marked with **"Drift Detected"**.
    - The report will show you exactly how the statistical distribution of that feature has changed (e.g., the mean of `sepal_length` shifted).
    - This is a powerful tool to automatically detect issues like the data poisoning you simulated.

---

## 📈 Result
A comprehensive, automated MLOps pipeline that not only trains and validates a model but also actively monitors for data quality issues, bias, and drift, while providing deep insights into model behavior. 🎯


