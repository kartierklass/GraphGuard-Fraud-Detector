# Jupyter Notebooks

This directory contains Jupyter notebooks for the GraphGuard fraud detection project.

## Planned Notebooks

### 1. `01_eda.ipynb` - Exploratory Data Analysis
- Data loading and inspection
- Feature distributions and correlations
- Missing value analysis
- Fraud rate analysis

### 2. `02_baseline_xgb.ipynb` - Baseline XGBoost Model
- Data preprocessing
- Feature engineering (tabular only)
- XGBoost training and tuning
- Baseline performance evaluation

### 3. `03_build_graph_node2vec.ipynb` - Graph Feature Engineering
- Graph construction from transaction data
- Node2Vec embedding computation
- Graph statistics calculation
- Feature extraction for transactions

### 4. `04_hybrid_train_eval.ipynb` - Hybrid Model Training
- Combined tabular + graph features
- Hybrid XGBoost training
- Performance comparison (baseline vs hybrid)
- Model evaluation and metrics

## Usage

1. **Install Jupyter**: `pip install jupyter`
2. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
3. **Navigate**: Open the notebooks in order for the complete pipeline

## Dependencies

The notebooks require the same dependencies as the main project. Install them with:

```bash
pip install -r requirements.txt
```

## Data Requirements

Before running the notebooks:
1. Download your chosen dataset (IEEE-CIS or PaySim)
2. Place it in `data/raw/`
3. Update the data loading paths in the notebooks

## Expected Outputs

- Model performance metrics and comparisons
- Feature importance analysis
- SHAP explanations and visualizations
- Graph visualizations and statistics

## Notes

- Notebooks are designed to be run sequentially
- Each notebook saves intermediate results for the next one
- Modify data paths and parameters as needed for your dataset
- Results and models are saved to `app/artifacts/`
