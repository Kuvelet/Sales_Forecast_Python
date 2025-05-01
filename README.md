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

In this section, we implement a modular forecasting pipeline using Facebook Prophet, a powerful time series model developed for business applications with seasonality and trend components. Prophet is particularly well-suited for datasets with clear temporal structure and missing data, and it allows for interpretable forecasting at scale. We train a separate model for each SKU using monthly sales data from January 2023 to August 2024, forecast the next 7 months, and clean the results for usability. Prophet’s strength lies in its ability to model trends automatically and generate forecasts that can adapt to the dynamic behavior of each product, making it ideal for demand planning and procurement decisions.

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

- This step splits the time series data into a training set and a test (evaluation) set to allow proper model validation and performance assessment.

- By reserving the most recent 7 months as a test set, we can simulate a real-world forecasting scenario, where future demand is unknown and needs to be predicted.


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

In this step, I build a monthly sales forecast for each SKU using Facebook Prophet, a time series model that captures seasonality and trend.

We train a separate model for each SKU, forecast 7 months into the future, and clean the results for interpretability.

What Below Code Does:

- Loops through each SKU and trains a standalone Prophet model on its monthly sales data.
- Forecasts 7 months ahead, using the .make_future_dataframe() method.
- Adds Item_ID to the forecast output so we can trace predictions back to the SKU.
- Handles any modeling errors gracefully with a try-except block.
- Cleans the output by:
  - Replacing negative predictions and tiny decimals (e.g. 8.88e-16) with 0.
  - Rounding all values to 2 decimal places for neatness.
- Exports the forecast to prophet_full_forecast.csv.


```python
# Step3A.4 Forecast Using Prophet

from prophet import Prophet
from tqdm import tqdm
import warnings


#Imports Python's built-in warnings module and suppresses warning messages. This cleans up the output by hiding minor warnings (common with Prophet).
warnings.filterwarnings("ignore")

# Store forecasts
prophet_forecasts_nofilter = []

# Forecast loop
for sku in tqdm(sku_list):
    sku_train_df = training_data[training_data['Item_ID'] == sku][['YearMonth', 'Monthly_Quantity']]

    # Skip SKUs with zero historical sales
    if sku_train_df['Monthly_Quantity'].sum() == 0:
        continue

    # Prepare data for Prophet
    prophet_df = sku_train_df.rename(columns={'YearMonth': 'ds', 'Monthly_Quantity': 'y'})

    try:
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=7, freq='M')
        forecast = model.predict(future)

        forecast['Item_ID'] = sku
        prophet_forecasts_nofilter.append(forecast)

    except Exception as e:
        print(f"SKU {sku} error: {e}")

# Combine results
prophet_forecast_df = pd.concat(prophet_forecasts_nofilter, ignore_index=True)
prophet_forecast_df.rename(columns={'ds': 'ForecastMonth', 'yhat': 'Forecasted_Quantity'}, inplace=True)

# Clean Prophet forecasts (no negative or near-zero values)
prophet_forecast_df['Forecasted_Quantity'] = prophet_forecast_df['Forecasted_Quantity'].apply(
    lambda x: 0 if x < 0 or abs(x) < 1e-5 else round(x, 2)
)

# Save forecast data
prophet_forecast_df.to_csv("prophet_full_forecast.csv", index=False)
```

#### Step3A.5 - Compare Forecast with Actuals

After generating SKU-level forecasts with Prophet, we compare those predictions to the actual observed sales. This comparison allows us to evaluate model accuracy and prepare for metric calculations such as RMSE, MAE, and MAPE.

What Below Code Does:

- Creates a new comparison DataFrame that aligns forecasted and actual values for each SKU and each month.
- Renames YearMonth to ForecastMonth so it can match Prophet’s output format and maintain clarity.
- Uses a left join so all actuals are preserved — even if no forecast was generated (e.g. due to skipped or failed SKUs).
- Fills missing forecasted values with 0, assuming the model didn’t produce a result for those entries.
- Adds a new column called Forecast_Error, representing the difference between forecasted and actual sales quantity.
- Saves the full comparison to a CSV file, which can be used for evaluation or visualization.

```python
# Step3A.5 Compare Forecast with Actuals

# Prepare DataFrames
actual_df = all_sku_monthly_w0.copy()
actual_df.rename(columns={'YearMonth': 'ForecastMonth', 'Monthly_Quantity': 'Actual'}, inplace=True)

forecast_df = prophet_forecast_df.copy()

# Merge Actuals LEFT JOIN Forecasts
comparison_df = pd.merge(
    actual_df[['ForecastMonth', 'Item_ID', 'Actual']],
    forecast_df[['ForecastMonth', 'Item_ID', 'Forecasted_Quantity']],
    on=['ForecastMonth', 'Item_ID'],
    how='left'  # Keep all actuals
)

# Fill missing Forecasted_Quantity with 0
comparison_df['Forecasted_Quantity'] = comparison_df['Forecasted_Quantity'].fillna(0)

# Calculate Forecast_Error
comparison_df['Forecast_Error'] = comparison_df['Forecasted_Quantity'] - comparison_df['Actual']

# Save the full comparison
comparison_df.to_csv("actual_vs_prophet_forecasts.csv", index=False)

print("✅ Full Forecast vs Actual (Full Lineup) Saved: 'actual_vs_prophet_forecasts.csv'")
```
#### Step3A.6 - Evaluate Prophet Forecast Accuracy

In this step, we compute standard forecast accuracy metrics — RMSE, MAE, and MAPE — by comparing the forecasted quantities with actual sales from the test period (2024-09-30 to 2025-03-31).
These metrics help quantify how well the Prophet model performed in a real-world scenario.

What Below Code Does:

- Loads the comparison file actual_vs_prophet_forecasts.csv created earlier.
- Converts the ForecastMonth column to datetime to enable time-based filtering.
- Converts the ForecastMonth column to datetime to enable time-based filtering.
- Computes three metrics:
  - RMSE (Root Mean Squared Error): Sensitive to large errors; penalizes big misses more.
  - MAE (Mean Absolute Error): Average absolute difference between prediction and reality.
  - MAPE (Mean Absolute Percentage Error): Expresses error as a % of the actual — but only calculated where actual ≠ 0 to avoid undefined results.
- Prints all metrics, rounded to 2 decimal places for reporting.

```python
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# Load and prepare forecast comparison file
comparison_df = pd.read_csv("actual_vs_prophet_forecasts.csv")
comparison_df['ForecastMonth'] = pd.to_datetime(comparison_df['ForecastMonth'])

# Filter to forecast horizon only
eval_df = comparison_df[comparison_df['ForecastMonth'] >= '2024-09-30']

# Compute metrics
rmse = root_mean_squared_error(eval_df['Actual'], eval_df['Forecasted_Quantity'])
mae = mean_absolute_error(eval_df['Actual'], eval_df['Forecasted_Quantity'])

# Avoid divide-by-zero issue in MAPE by filtering out zero-actual rows
nonzero_eval_df = eval_df[eval_df['Actual'] != 0]
mape = mean_absolute_percentage_error(
    nonzero_eval_df['Actual'],
    nonzero_eval_df['Forecasted_Quantity']
) * 100

# Display results
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
```
### Wrap-Up: Prophet Forecasting Approach

In this section, we implemented a robust forecasting pipeline using Facebook Prophet, tailored for monthly SKU-level demand prediction. We trained individual models for over 3,000 SKUs using sales data from January 2023 to August 2024, then generated 7-month ahead forecasts through March 2025. After aligning the forecast output with actual sales data, we evaluated performance using standard accuracy metrics — RMSE, MAE, and MAPE. Additionally, we refined Prophet’s output by eliminating negative and near-zero predictions, ensuring the results were both interpretable and production-ready. This forecasting framework provides a scalable and transparent foundation for inventory planning and supplier ordering decisions.

## STEP 3B - ARIMA Model

In this section, we implement a second forecasting approach using ARIMA (AutoRegressive Integrated Moving Average) - powerful statistical model for time series forecasting. Unlike Prophet, ARIMA does not assume predefined seasonal patterns or trend components. Instead, it learns temporal dependencies from past sales data and is well-suited for datasets with consistent time intervals and moderate history length. We train a separate ARIMA model for each SKU using historical monthly sales, forecast the next 7 months, and apply post-processing steps to clean and store the results. This section enables a direct, head-to-head comparison with Prophet to determine which modeling technique better captures sales behavior in our dataset.

### Step 3B.1 – Import Required Libraries for ARIMA Forecasting

Before we begin building ARIMA-based forecasts, we import the necessary Python libraries for data manipulation, iteration, and silent error handling. These are foundational tools used throughout the modeling pipeline.

What Below Code Does:

- pandas, The core library for handling structured data. I use it to load CSV files, manipulate time series, and structure forecast outputs.
- numpy,  Enables efficient numerical operations, such as filling or transforming arrays and computing statistical measures.
- tqdm, Provides a real-time progress bar in loops, especially useful for tracking the per-SKU forecasting loop over thousands of SKUs.
- warnings.filterwarnings("ignore"), Silences non-critical warnings (often thrown by ARIMA internals), keeping output clean and focused during batch processing.

```python
# Load the monthly SKU data including zero-sales months
all_sku_monthly_w0 = pd.read_csv("all_sku_monthly_w0.csv")

# Ensure correct date format
all_sku_monthly_w0['YearMonth'] = pd.to_datetime(all_sku_monthly_w0['YearMonth'])
```

### Step 3B.2 – Load the data

To build reliable ARIMA models, we start by loading the preprocessed dataset that includes every SKU for every month, even those with zero sales. Ensuring complete time series continuity is crucial for ARIMA, which assumes a fixed time interval between observations.This step ensures we have a clean, gap-free, time-indexed dataset before feeding it into ARIMA.

What Below Code Does:
- Reads the preprocessed file all_sku_monthly_w0.csv, which contains a full grid of monthly sales values for each SKU, including months where no units were sold (represented as zeros).
- Converts the YearMonth column into a proper datetime format using pd.to_datetime(). This is critical for ARIMA, which requires time series indices to be uniform and date-aware.

```python
# Step3B.2 - Load the data

# Load the monthly SKU data including zero-sales months
all_sku_monthly_w0 = pd.read_csv("all_sku_monthly_w0.csv")

# Ensure correct date format
all_sku_monthly_w0['YearMonth'] = pd.to_datetime(all_sku_monthly_w0['YearMonth'])
```

### Step 3B.3 – Define Training and Test Sets

To evaluate the forecasting accuracy of our ARIMA models, we split the time series data into training and test sets. Each SKU’s historical sales are divided chronologically, with earlier months used to train the model and later months used to test the forecasted values against actual outcomes.

What Below Code Does:
- The use of end-of-month-aligned dates (e.g., 2023-01-31) ensures compatibility with Prophet’s outputs and avoids double entries from mismatched date formats.
- Training Set (January 2023 – August 2024): This period provides up to 20 months of historical data per SKU, giving the ARIMA model enough information to learn patterns in demand.
- Test Set (September 2024 – March 2025):This 7-month window represents our forecast horizon. Predictions made by ARIMA will be compared against these real sales figures for evaluation.

```python
# Step3B.3 - Define Training and Test Sets

# Define training set
training_data = all_sku_monthly_w0[
    (all_sku_monthly_w0['YearMonth'] >= '2023-01-31') & 
    (all_sku_monthly_w0['YearMonth'] <= '2024-08-31')
]

# Define test set (forecast period)
test_data = all_sku_monthly_w0[
    (all_sku_monthly_w0['YearMonth'] >= '2024-09-30') & 
    (all_sku_monthly_w0['YearMonth'] <= '2025-03-31')
]
```

### Step 3B.4 – Install and Import ARIMA

We use the pmdarima library to automatically fit ARIMA models for each SKU. This library provides the auto_arima() function, which simplifies the model selection process by testing multiple combinations of ARIMA parameters and choosing the best one based on information criteria

What Below Code Does:
- !pip install pmdarima installs the pmdarima package, which is a Python wrapper around the well-known statsmodels ARIMA implementation. It includes automatic hyperparameter tuning and seasonal decomposition.
- from pmdarima import auto_arima imports the core function used to build and fit ARIMA models for each SKU.

```python
# Install the pmdarima package (one-time installation)
!pip install pmdarima

# Import the auto_arima function from pmdarima
from pmdarima import auto_arima
```

### Step 3B.5 – Set Up Forecasting Loop with ARIMA

In this step, we loop through each unique SKU in the training set, fit a dedicated ARIMA model for each one, and forecast sales for the next 7 months. This logic enables independent forecasting per SKU and includes built-in fallbacks to ensure no product is left behind due to model fitting errors.

What Below Code Does:
- SKU-by-SKU Training Loop:
  - Iterates through each unique SKU, filters relevant training data, and fits an individual ARIMA model.
- Model Fitting:
  - Uses auto_arima() with seasonal=False to allow the model to find the best non-seasonal (p,d,q) configuration using stepwise optimization. This keeps it lightweight and avoids overfitting.
- Fallback Handling:
  - If ARIMA fails to converge or throws an exception (common for sparse data), a fallback mechanism replaces the forecast with a flat average of the last 3 months.
- Forecast Post-Processing: Forecasts are sanitized to ensure
  - Negative values are set to 0
  - Extremely small values (e.g., 1e-16) are also zeroed
  - Values are rounded to 2 decimals for readability
- Final Output:
  - All forecasts are stored in a single DataFrame and exported to arima_full_forecast.csv for later evaluation.

 ```python
# Step3B.5 - Set up forecasting loop

# Prepare list to store forecasts
arima_forecasts = []

# Get unique SKUs
sku_list = training_data['Item_ID'].unique()

# Forecasting loop
for sku in tqdm(sku_list):
    # Filter the training data for the current SKU
    sku_train_df = training_data[training_data['Item_ID'] == sku][['YearMonth', 'Monthly_Quantity']]

    # Skip SKUs with all zero historical sales
    if sku_train_df['Monthly_Quantity'].sum() == 0:
        continue

    # Ensure datetime index
    sku_train_df['YearMonth'] = pd.to_datetime(sku_train_df['YearMonth'])
    sku_train_df.set_index('YearMonth', inplace=True)

    # Extract the target Series for ARIMA
    y = sku_train_df['Monthly_Quantity']

    # Sanity check
    if y.isnull().any():
        print(f"SKU {sku} error: NaNs detected even after cleaning — skipping")
        continue

    try:
        # Fit ARIMA without seasonality
        model = auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True
        )

        # Forecast next 7 months
        forecast = model.predict(n_periods=7)

    except Exception as e:
        print(f"SKU {sku} error: {e} — using fallback average")

        # Fallback: use 3-month average if ARIMA fails
        fallback_value = y.tail(3).mean()
        forecast = [fallback_value] * 7

    # Store forecast results
    forecast_df = pd.DataFrame({
        'ForecastMonth': pd.date_range(start='2024-09-30', periods=7, freq='M'),
        'Forecasted_Quantity': forecast,
        'Item_ID': sku
    })

    arima_forecasts.append(forecast_df)

# Combine forecasts into single DataFrame
arima_forecast_df = pd.concat(arima_forecasts, ignore_index=True)

# Fix negative and scientific notation values
arima_forecast_df['Forecasted_Quantity'] = arima_forecast_df['Forecasted_Quantity'].apply(
    lambda x: 0 if x < 0 or abs(x) < 1e-5 else round(x, 2)
)

# Save forecasts to CSV
arima_forecast_df.to_csv("arima_full_forecast.csv", index=False)

# Preview
arima_forecast_df.head()
```

### Step 3B.6 – Compare ARIMA Forecasts with Actuals

Once we've generated forecasts for all SKUs, we evaluate model performance by comparing predictions against the actual sales data from the test period. This step ensures we measure how well the ARIMA model performs in real-world forecasting.

What Below Code Does:
- Creates a test set by filtering only the actuals from the 7-month forecast window.
- Standardizes the column names (ForecastMonth, Actual) to match the forecast DataFrame and enable clean merging.
- Merges forecasted values with actual sales using a LEFT JOIN, ensuring that even SKUs without forecasted output are preserved for analysis.
- Fills missing forecasts with 0 — this reflects a conservative assumption that no prediction is made for those months.
- Calculates Forecast Error as the difference between forecasted and actual sales.
- Saves the result into a CSV for later metric evaluation and reporting.

```python
## Step3B.6 - Compare Forecast with Actuals
actual_df = all_sku_monthly_w0[
    (all_sku_monthly_w0['YearMonth'] >= '2024-09-30') & 
    (all_sku_monthly_w0['YearMonth'] <= '2025-03-31')].copy()

actual_df.rename(columns={'YearMonth': 'ForecastMonth', 'Monthly_Quantity': 'Actual'}, inplace=True)

forecast_df = arima_forecast_df.copy()

# Merge Actuals LEFT JOIN Forecasts
comparison_df = pd.merge(
    actual_df[['ForecastMonth', 'Item_ID', 'Actual']],
    forecast_df[['ForecastMonth', 'Item_ID', 'Forecasted_Quantity']],
    on=['ForecastMonth', 'Item_ID'],
    how='left'  # Keep all actuals
)

# Fill missing Forecasted_Quantity with 0
comparison_df['Forecasted_Quantity'] = comparison_df['Forecasted_Quantity'].fillna(0)

# (Optional) Calculate Forecast_Error
comparison_df['Forecast_Error'] = comparison_df['Forecasted_Quantity'] - comparison_df['Actual']

# Save the full comparison
comparison_df.to_csv("actual_vs_arima_forecasts.csv", index=False)

print("✅ Full Forecast vs Actual (Full Lineup) Saved: 'Actual_vs_arima_forecasts.csv'")
```
### Step 3B.7 – Evaluate ARIMA Forecast Accuracy

To quantify how well ARIMA forecasts performed, we compute standard error metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). These help assess the deviation between predicted and actual sales for the test period.

What Below Code Does:
- Loads the forecast vs actual CSV generated in the previous step and parses the ForecastMonth column into datetime format.
- Filters the dataset to include only the forecast horizon: September 2024 to March 2025.
- Calculates:
  - RMSE: Measures average magnitude of error — more sensitive to large errors.
  - MAE: Measures average absolute deviation — more interpretable and stable.
  - MAPE: Calculates average percentage error — ideal for interpreting accuracy as a percentage of actual demand.
- Handles division-by-zero by excluding rows where actual quantity is zero from the MAPE calculation.
- Prints the results in a clean and interpretable format.

### Wrap-Up: ARIMA Forecasting Approach

The ARIMA (AutoRegressive Integrated Moving Average) model was used to forecast monthly sales quantities for over 3,000 unique SKUs. By individually fitting each product's time series using auto_arima() from the pmdarima library, we allowed the model to intelligently select optimal parameters while excluding seasonality to maintain generality.

To ensure completeness and robustness:
- Any SKUs that failed to converge due to statistical instability were automatically handled using a fallback forecast based on the 3-month trailing average.
- Forecasts were post-processed to prevent negative or near-zero outputs, ensuring all predicted quantities were production-ready.
- Evaluation was performed on a 7-month forecast horizon (September 2024 to March 2025), comparing predictions to actual sales using RMSE, MAE, and MAPE.

These metrics allowed us to objectively compare ARIMA against other models (such as Prophet) and determine the most reliable forecasting strategy for procurement planning.
