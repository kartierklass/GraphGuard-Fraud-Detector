# Data Directory

This directory contains the datasets used for training and testing the GraphGuard fraud detection model.

## Directory Structure

```
data/
├── raw/           # Raw dataset files (CSV, Parquet, etc.)
├── processed/     # Preprocessed and feature-engineered data
└── README.md      # This file
```

## Dataset Requirements

The GraphGuard model expects transaction data with the following minimal schema:

- `transaction_id`: Unique identifier for each transaction
- `amount`: Transaction amount (numeric)
- `src_account_id`: Source account identifier
- `dst_account_id`: Destination account identifier
- `device_id`: Device identifier (optional)
- `ip_address`: IP address (optional)
- `merchant_id`: Merchant identifier (optional)
- `timestamp`: Transaction timestamp (optional)
- `label`: Fraud label (0/1 or True/False)

## Recommended Datasets

### 1. IEEE-CIS Fraud Detection (Preferred)
- **Source**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: ~6GB
- **Features**: Rich e-commerce transaction data with device, card, and identity information
- **Realism**: High - based on real e-commerce transactions

### 2. PaySim (Alternative)
- **Source**: [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Size**: ~1GB
- **Features**: Synthetic mobile money transactions
- **Realism**: Medium - synthetic but realistic patterns

## Data Preparation Steps

1. **Download Dataset**: Choose one of the recommended datasets above
2. **Place in Raw Directory**: Put the raw files in `data/raw/`
3. **Update Schema Mapping**: Modify `src/preprocess.py` to map your dataset columns to the expected schema
4. **Run Preprocessing**: Execute the preprocessing pipeline

## Example Data Loading

```python
# In src/preprocess.py
def load_data(self, file_path: str) -> pd.DataFrame:
    """Load transaction data from CSV/Parquet file"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    # Map columns to expected schema
    # Example for IEEE-CIS dataset:
    column_mapping = {
        'TransactionID': 'transaction_id',
        'TransactionAmt': 'amount',
        'card1': 'src_account_id',  # Simplified mapping
        'card2': 'dst_account_id',  # Simplified mapping
        'DeviceType': 'device_id',
        'id_01': 'ip_address',      # Simplified mapping
        'ProductCD': 'merchant_id',  # Simplified mapping
        'TransactionDT': 'timestamp',
        'isFraud': 'label'
    }
    
    df = df.rename(columns=column_mapping)
    return df
```

## Data Privacy & Security

- **Never commit raw data files** to version control
- **Use .gitignore** to exclude data files
- **Respect dataset licenses** and terms of use
- **Anonymize sensitive data** if required for your use case

## Next Steps

1. Download your chosen dataset
2. Place it in `data/raw/`
3. Update the preprocessing code to handle your specific dataset format
4. Run the training pipeline: `make train`
