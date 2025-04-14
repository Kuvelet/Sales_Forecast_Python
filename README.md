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

I start by importing Python libraries required for data analysis and cleaning.

```python
# Data manipulation
import pandas as pd
import numpy as np

# Date handling
from datetime import datetime

# Display options
pd.set_option('display.max_columns', None)
```





