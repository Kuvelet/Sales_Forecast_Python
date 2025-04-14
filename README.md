# Auto Parts Sales Forecasting Project

## Overview
Forecasting monthly auto part SKU sales using historical sales data (2023 onwards).

## Project Structure
- `data/`: Contains datasets used and created.
- `notebooks/`: Contains Jupyter Notebooks for analysis and forecasting.

## Steps:
1. Data Cleaning & Preparation
2. Exploratory Data Analysis (EDA)
3. Model Training (ARIMA, Holt-Winters, Prophet)
4. SKU Group Forecasting
5. Model Accuracy Comparison
6. Final SKU-level Forecast

## Forecasting Approach:
- Hybrid approach: Individual forecasts for top 500 SKUs, group forecasting for the remaining SKUs.

## Tools Used:
- Python (Pandas, NumPy, Statsmodels, Prophet)
- Jupyter Notebook
- GitHub (for documentation & collaboration)

### 1) Data Cleaning & Preparation

Step 1.1: Import Necessary Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Date handling
from datetime import datetime

# Display options
pd.set_option('display.max_columns', None)
```

Step 1.2: Load the Data from CSV

```python
# Load data from your CSV file
data = pd.read_csv('your_sales_data.csv')

# Preview first 5 rows
data.head()
```

Step 1.3: Initial Data Cleaning

Remove irrelevant rows as per your criteria:

- Remove rows where Quantity is 0 or null.
- Remove rows where Item ID is null.

```python

# Check initial row count : The f prefix means “formatted string”.Everything inside the {} is evaluated as Python code, and its result is inserted into the string.data.shape[0] gives just the row count.

print(f"Initial Rows: {data.shape[0]}")

# Remove rows with Quantity = 0 or Quantity is null
data_clean = data[data['Quantity'].notnull() & (data['Quantity'] != 0)]

# Remove rows with null Item ID
data_clean = data_clean[data_clean['Item ID'].notnull()]

# Check rows after cleaning
print(f"Rows after cleaning: {data_clean.shape[0]}")
```

