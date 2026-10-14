from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "01_Data" / "raw_sales.csv"
OUTPUT_DIR = BASE_DIR / "04_Output"
SCREENSHOTS_DIR = BASE_DIR / "03_Screenshots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATA INGESTION AND QUALITY CHECKS
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDATA QUALITY CHECKS")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing Order Dates:", df["Order Date"].isna().sum())
print("Missing Sales:", df["Sales"].isna().sum())

df["Order Date"] = pd.to_datetime(
    df["Order Date"], dayfirst=True, errors="coerce"
)
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

invalid_dates = df["Order Date"].isna().sum()
invalid_sales = df["Sales"].isna().sum()

print("Invalid or missing dates:", invalid_dates)
print("Invalid or missing sales:", invalid_sales)

if invalid_dates or invalid_sales:
    raise ValueError("Invalid dates or sales values detected.")

# Each row represents an order line; repeated orders can be valid.


# ============================================================
# 3. MONTHLY SALES AGGREGATION
# ============================================================

monthly_sales = (
    df.set_index("Order Date")["Sales"]
    .resample("MS")
    .sum(min_count=1)
    .asfreq("MS")
)

if monthly_sales.isna().any():
    raise ValueError("Missing monthly sales periods detected.")

if len(monthly_sales) < 36:
    raise ValueError("At least 36 monthly observations are required.")

print("\nTIME SERIES SUMMARY")
print("Monthly observations:", len(monthly_sales))
print("First month:", monthly_sales.index.min().date())
print("Last month:", monthly_sales.index.max().date())


# ============================================================
# 4. BUSINESS KPI ANALYSIS
# ============================================================

print("\nBUSINESS KPIs")
print(f"Total Revenue: ${monthly_sales.sum():,.2f}")
print(f"Average Monthly Revenue: ${monthly_sales.mean():,.2f}")
print(f"Highest Revenue Month: {monthly_sales.idxmax():%Y-%m}")
print(f"Lowest Revenue Month: {monthly_sales.idxmin():%Y-%m}")

annual_sales = monthly_sales.resample("YS").sum()
annual_growth = annual_sales.pct_change() * 100

print("\nANNUAL REVENUE")
for date, revenue in annual_sales.items():
    growth = annual_growth.loc[date]
    growth_text = (
        f"{growth:.2f}%"
        if pd.notna(growth)
        else "N/A"
    )
    print(f"{date.year}: ${revenue:,.2f} | YoY Growth: {growth_text}")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

# Reserve the final 12 months for out-of-sample evaluation.

train = monthly_sales.iloc[:-12]
test = monthly_sales.iloc[-12:]

print("\nTRAIN / TEST SPLIT")
print(f"Training: {train.index.min():%Y-%m} to {train.index.max():%Y-%m}")
print(f"Testing:  {test.index.min():%Y-%m} to {test.index.max():%Y-%m}")


# ============================================================
# 6. MODEL EVALUATION FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(
        mean_squared_error(actual_values, predicted_values)
    )

    nonzero = actual_values != 0
    mape = (
        np.mean(
            np.abs(
                (actual_values[nonzero] - predicted_values[nonzero])
                / actual_values[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }


def fit_holt_winters(series):
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    )
    return model.fit(optimized=True)


# ============================================================
# 7. HOLT-WINTERS MODEL
# ============================================================

hw_fit = fit_holt_winters(train)
hw_predictions = hw_fit.forecast(len(test))
hw_predictions.index = test.index


# ============================================================
# 8. SEASONAL NAIVE BASELINE
# ============================================================

# Predict each test month using the corresponding month
# from the previous year.

naive_predictions = pd.Series(
    train.iloc[-12:].to_numpy(),
    index=test.index,
    name="Seasonal_Naive"
)


# ============================================================
# 9. MODEL PERFORMANCE COMPARISON
# ============================================================

hw_metrics = calculate_metrics(test, hw_predictions)
naive_metrics = calculate_metrics(test, naive_predictions)

metrics_df = pd.DataFrame(
    [
        {"Model": "Holt-Winters", **hw_metrics},
        {"Model": "Seasonal Naive", **naive_metrics}
    ]
)

print("\nMODEL PERFORMANCE COMPARISON")
print(metrics_df.round(2).to_string(index=False))

metrics_df.to_csv(
    OUTPUT_DIR / "model_evaluation_metrics.csv",
    index=False
)

evaluation_df = pd.DataFrame({
    "Month": test.index,
    "Actual_Sales": test.to_numpy(),
    "Holt_Winters_Prediction": hw_predictions.to_numpy(),
    "Seasonal_Naive_Prediction": naive_predictions.to_numpy()
})

evaluation_df.to_csv(
    OUTPUT_DIR / "model_evaluation_predictions.csv",
    index=False
)


# ============================================================
# 10. MODEL EVALUATION VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    test.index, test.values,
    label="Actual Revenue", marker="o", linewidth=2
)
plt.plot(
    test.index, hw_predictions.values,
    label="Holt-Winters", marker="o", linestyle="--"
)
plt.plot(
    test.index, naive_predictions.values,
    label="Seasonal Naive", marker="s", linestyle=":"
)

plt.title("Sales Forecast Model Evaluation | Test Period")
plt.xlabel("Month")
plt.ylabel("Monthly Revenue ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "01_model_evaluation.png",
    dpi=300
)
plt.show()


# ============================================================
# 11. FINAL 12-MONTH FORECAST
# ============================================================

# Refit on all historical observations after evaluation.

final_fit = fit_holt_winters(monthly_sales)
forecast = final_fit.forecast(12)

forecast_df = pd.DataFrame({
    "Forecast_Month": forecast.index,
    "Projected_Sales": forecast.values
})

forecast_df.to_csv(
    OUTPUT_DIR / "forecast_values.csv",
    index=False
)

print("\nFINAL 12-MONTH FORECAST")
print(forecast_df.to_string(index=False))
print(f"\nTotal Forecast Revenue: ${forecast.sum():,.2f}")


# ============================================================
# 12. FINAL FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    label="Historical Monthly Revenue",
    linewidth=2
)

plt.plot(
    forecast.index,
    forecast.values,
    label="Holt-Winters Forecast",
    linestyle="--",
    linewidth=2
)

plt.axvline(
    monthly_sales.index[-1],
    color="gray",
    linestyle=":",
    label="Forecast Boundary"
)

plt.title("Monthly Sales Revenue | Historical Data and 12-Month Forecast")
plt.xlabel("Month")
plt.ylabel("Monthly Revenue ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    SCREENSHOTS_DIR / "02_final_sales_forecast.png",
    dpi=300
)

# Also update the existing screenshot used by the README.
plt.savefig(
    SCREENSHOTS_DIR / "sales_forecast_plot.png",
    dpi=300
)

plt.show()

print("\nSALES FORECASTING ANALYSIS COMPLETE")
print("Evaluation results and forecast CSV files exported.")
print("Visualizations saved to 03_Screenshots.")
