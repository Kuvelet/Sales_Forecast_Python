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

## STEP 1 - Data Cleaning & Preparation

### Step 1.1: Import Necessary Libraries

```python
# Step 1.1
# Import Libraries
import pandas as pd
import numpy as np

# To handle dates appropriately
from datetime import datetime

# Display options
pd.set_option('display.max_columns', None)

```

### Step 1.2: Load the Data from CSV

```python
# Step 1.2
# Load data from your CSV file
data = pd.read_csv(r'V:\DATA\SO_FORECAST.CSV', encoding='ISO-8859-1')

# Preview first 5 rows
data.head()

```

### Step 1.3: Initial Data Cleaning

Remove irrelevant rows as per your criteria:

- Remove rows where Quantity is 0 or null.
- Remove rows where Item ID is null.

```python
# Step 1.3
# Check initial row count : The f prefix means “formatted string”.Everything inside the {} is evaluated as Python code, and its result is inserted into the string.data.shape[0] gives just the row count.

print(f"Initial Rows: {data.shape[0]}")

# Remove rows with Quantity = 0 or Quantity is null : Boolean indexing — selecting rows where a certain condition is True

data_clean = data[data['Quantity'].notnull() & (data['Quantity'] != 0)]

# Remove rows with null Item ID
data_clean = data_clean[data_clean['Item ID'].notnull()]

# Check rows after cleaning
print(f"Rows after cleaning: {data_clean.shape[0]}")
```
### Step 1.4: Filter Data from 2023 onward

```python
# Step 1.4
# Convert 'Date' to datetime
data_clean['Date'] = pd.to_datetime(data_clean['Date'], errors='coerce')

# Filter data from 2023 onwards
data_2023 = data_clean[data_clean['Date'] >= '2023-01-01'].copy()

# Check rows after filtering
print(f"Rows from 2023 onwards: {data_2023.shape[0]}")
```

### Step 1.5: Aggregate Monthly Sales per SKU

```python
# Step 1.5
# Create Year-Month column
data_2023['YearMonth'] = data_2023['Date'].dt.to_period('M')

# Aggregate monthly quantities per SKU
monthly_sku_data = data_2023.groupby(['YearMonth', 'Item ID']).agg({
    'Quantity': 'sum'
}).reset_index()

# Rename columns clearly
monthly_sku_data.columns = ['YearMonth', 'Item_ID', 'Monthly_Quantity']

# Check aggregated data
monthly_sku_data.head()
```

### Step 1.6: Identify Top SKUs

```python
# Step 1.6
# Get total quantities per SKU to identify top SKUs
total_qty_per_sku = monthly_sku_data.groupby('Item_ID')['Monthly_Quantity'].sum().reset_index()

# Sort SKUs by descending quantity
top_skus = total_qty_per_sku.sort_values(by='Monthly_Quantity', ascending=False).head(500)
top_skus = top_skus.rename(columns={'Monthly_Quantity': 'Total Sales Quantity'})

# Check top SKUs
top_skus.head(10)
```

### Step 1.7: Seperate monthly_sku_data

```python
# Step 1.7
# Top 500 SKUs data
top_sku_data = monthly_sku_data[monthly_sku_data['Item_ID'].isin(top_skus['Item_ID'])]

# Remaining SKUs data
remaining_sku_data = monthly_sku_data[~monthly_sku_data['Item_ID'].isin(top_skus['Item_ID'])]

# Preview data
print(f"Top SKUs data rows: {top_sku_data.shape[0]}")
print(f"Remaining SKUs data rows: {remaining_sku_data.shape[0]}")
print(f"All SKUs data rows: {monthly_sku_data.shape[0]}")
```

### Step 1.8: Export Datasets

```python
# Step 1.8
# Export datasets
top_sku_data.to_csv('top_sku_monthly.csv', index=False)
remaining_sku_data.to_csv('remaining_sku_monthly.csv', index=False)
monthly_sku_data.to_csv('all_sku_monthly.csv', index=False)
```

In this section, we performed essential preprocessing to ensure our dataset was clean and ready for reliable forecasting. The key actions included:

- **Removed invalid entries**: Rows with `null` or `0` quantities were filtered out to avoid skewing the results.
- **Ensured SKU validity**: Entries without a valid `Item ID` were excluded, as each SKU must be uniquely identifiable.
- **Filtered time range**: Data was limited to entries from January 1, 2023, onward to focus on recent sales patterns.
- **Grouped and aggregated**: Cleaned daily data was aggregated monthly per SKU to simplify modeling and reduce noise.
- **Segmented dataset**: SKUs were divided into two categories:
  - `top_sku_data`: The top 500 SKUs by total quantity sold.
  - `remaining_sku_data`: All other SKUs.
- **Exported datasets**: Cleaned and grouped data was exported into three separate CSV files for streamlined access:
  - `top_sku_monthly.csv`
  - `remaining_sku_monthly.csv`
  - `all_sku_monthly.csv`

This structured approach improves model performance by ensuring consistent and high-quality input data.

---

## STEP 2 - Exploratory Data Analysis (EDA)

### Step 2.1: Import Data & Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned monthly SKU data
monthly_data = pd.read_csv('monthly_sku_data.csv')

# Convert 'YearMonth' to datetime
monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'].astype(str))
```

### Step 2.2: Overview of SKU Distribution

```pyton
# Step 2.2
# Number of unique SKUs
print("Unique SKUs:", monthly_data['Item_ID'].nunique())

# Monthly time range
print("Date Range:", monthly_data['YearMonth'].min(), "to", monthly_data['YearMonth'].max())

# Plot: Monthly sales volume trend (Total)
monthly_trend = monthly_data.groupby('YearMonth')['Monthly_Quantity'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_trend, x='YearMonth', y='Monthly_Quantity')
plt.title("Total Monthly Received SO's")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.grid(True)

plt.tight_layout()
plt.savefig("monthly_so_trend.jpg", format='jpg', dpi=300)

plt.show()
```
![monthly_so_trend](monthly_so_trend.jpg)

### Step 2.3: Sales Distribution by SKU

```python
# Step 2.3: Sales Distribution by SKU
# Total sales per SKU
sku_totals = monthly_data.groupby('Item_ID')['Monthly_Quantity'].sum().sort_values(ascending=False)

# Convert index (Item_IDs) to string for better y-axis labels
top_20 = sku_totals.head(20)
top_20.index = top_20.index.astype(str)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_20.values, y=top_20.index)
plt.title("Top 20 SKUs by Total SO Received")
plt.xlabel("Total Quantity")
plt.ylabel("Item ID")
plt.grid(True)
plt.show()
```
![top20_skus_total_so](top20_skus_total_so.jpg)

### Step 2.4: Individual SKU Seasonality Check

```python

# Step 2.4: Individual SKU Seasonality Check
import os
export_folder = "sku_charts"
os.makedirs(export_folder, exist_ok=True)

sample_skus = monthly_data['Item_ID'].value_counts().head(3).index.tolist()

for sku in sample_skus:
    temp = monthly_data[monthly_data['Item_ID'] == sku]
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=temp, x='YearMonth', y='Monthly_Quantity')
    plt.title(f'SKU {sku} - Monthly Sales Trend')
    plt.xlabel("Month")
    plt.ylabel("Quantity")
    plt.grid(True)

# Export before showing
    filename = f"{export_folder}/sku_{sku}_sales_trend.jpg"
    plt.tight_layout()
    plt.savefig(filename, format='jpg', dpi=300)
    
    plt.show()
```
![sku_8512191_so_trend](sku_8512191_so_trend.jpg)
![sku_8512323_so_trend](sku_8512323_so_trend.jpg)
![sku_8514156_so_trend](sku_8514156_so_trend.jpg)

## STEP 3A - Prophet Model

This section demonstrates a clean and scalable forecasting pipeline using Facebook Prophet to generate monthly demand forecasts for individual automotive part SKUs. The pipeline ensures full alignment between historical sales data and Prophet forecast outputs through end-of-month date normalization.

### Dataset Overview

- Source: Internal daily sales order data (2015–2025)
- Preprocessed To: Monthly aggregated SKU-level quantities
- Date Field Used: YearMonth
- Forecast Target: Monthly_Quantity per SKU

### Key Design Decisions

#### 1) End-of-Month (MonthEnd) Alignment

Prophet outputs forecast dates at the end of each month (e.g. 2023-01-31, 2023-02-28).

To ensure all joins, evaluations, and comparisons work without mismatch:

- All YearMonth values were explicitly converted using:

```python
monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'].astype(str)) + pd.offsets.MonthEnd(0)
```
This ensures that every timestamp reflects the last day of each month.

- The SKU-date grid (full_date_range) was generated using:

```python
full_date_range = pd.date_range(start='2023-01-01', end='2025-03-31', freq='ME')
```
The freq='ME' option ensures all generated dates are end-of-month values, matching Prophet's output.

#### 2) Training/Test Split Logic

| Type        | Date Range                    |
|-------------|-------------------------------|
| Training    | 2023-01-31 to 2024-08-31       |
| Forecasting | 2024-09-30 to 2025-03-31       |

Prophet is fit on training data and asked to predict the next 7 months. There is no overlap between training and forecast periods, ensuring honest evaluation.

#### 3) Forecast Granularity

- Each SKU is trained and forecasted individually using Prophet.
- Forecasts are stored as a combined DataFrame with Item_ID, ForecastMonth, and Forecasted_Quantity.

### Facebook Prophet Pipeline Steps with Explanations

#### Step 3A.1 - Load Libraries and Normalize Data

- Loads the preprocessed monthly sales data from CSV.
- Converts all date values in the YearMonth column to the last day of each month (e.g. 2023-01-31)

```python
import pandas as pd
import numpy as np
from itertools import product

# Load your existing monthly SKU data
monthly_data = pd.read_csv("all_sku_monthly.csv")
# Convert to end of month to align with Prophet forecast output
monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'].astype(str)) + pd.offsets.MonthEnd(0)
```

#### Step 3A.2 - Create Complete SKU-Month Grid

- Extracts all unique SKUs.
- Creates a list of all month-end dates from Jan 2023 to Mar 2025.
- Uses product() to build a DataFrame with all possible combinations of SKU and date.
- Merges the original sales data into the full SKU-month grid.
- Fills in missing sales values with 0, so we can forecast even when there's no recorded sales.

```python

full_date_range = pd.date_range(start='2023-01-01', end='2025-03-31', freq='ME')

sku_list = monthly_data['Item_ID'].unique()

sku_month_combinations = pd.DataFrame(list(product(sku_list, full_date_range)), columns=['Item_ID', 'YearMonth'])

all_sku_monthly_w0 = pd.merge(sku_month_combinations, monthly_data, on=['Item_ID', 'YearMonth'], how='left')

all_sku_monthly_w0['Monthly_Quantity'] = all_sku_monthly_w0['Monthly_Quantity'].fillna(0)

all_sku_monthly_w0.to_csv("all_sku_monthly_w0.csv", index=False)

all_sku_monthly_w0.head()
```

#### Step3A.3 - Set Training and Test Set

-This step splits the time series data into a training set and a test (evaluation) set to allow proper model validation and performance assessment.

-By reserving the most recent 7 months as a test set, we can simulate a real-world forecasting scenario, where future demand is unknown and needs to be predicted.


```python
# Training set (historical)
training_data = all_sku_monthly_w0[
    (all_sku_monthly_w0['YearMonth'] >= '2023-01-31') & 
    (all_sku_monthly_w0['YearMonth'] <= '2024-08-31')
]

# Test set (evaluation)
test_data = all_sku_monthly_w0[
    (all_sku_monthly_w0['YearMonth'] >= '2024-09-30') & 
    (all_sku_monthly_w0['YearMonth'] <= '2025-03-31')
]
```
#### Step3A.4 - Prophet Forecasting Loop

This block trains a separate Prophet model for each SKU using the historical training set defined earlier. It then forecasts monthly demand for the next 7 months and stores the results.

```python
from prophet import Prophet
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
```
- Prophet: Facebook's open-source time series forecasting library.
- tqdm: Adds a progress bar to the loop, useful for tracking large batch forecasting.
- warnings.filterwarnings("ignore"): Suppresses output clutter from benign warnings during model fitting.

```python
prophet_forecasts_nofilter = []
```

- Initializes a list to hold forecast results from each SKU.

