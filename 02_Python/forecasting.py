import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# 1. DATA INGESTION
# Loading historical sales data from the 01_Data directory
try:
    df = pd.read_csv('../01_Data/raw_sales.csv')
except FileNotFoundError:
    print("Error: 'raw_sales.csv' not found in 01_Data folder.")
    exit()

# 2. DATA PRE-PROCESSING
# Ensure 'Order Date' is in datetime format and handle day-first format
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df = df.sort_values('Order Date')

# 3. TIME SERIES AGGREGATION
# Resampling daily data to Monthly frequency to identify long-term trends
df['Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

# Convert Period back to Datetime for statistical modeling compatibility
monthly_sales['Month'] = monthly_sales['Month'].astype(str)
monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
monthly_sales.set_index('Month', inplace=True)

# 4. BUSINESS KPI ANALYSIS
# Extracting key metrics to evaluate historical performance
total_sales = monthly_sales['Sales'].sum()
avg_sales = monthly_sales['Sales'].mean()
max_month = monthly_sales['Sales'].idxmax()
min_month = monthly_sales['Sales'].idxmin()

print("\n📊 BUSINESS KPIs SUMMARY:")
print(f"Total Revenue Analyzed: {total_sales:,.2f}")
print(f"Average Monthly Revenue: {avg_sales:,.2f}")
print(f"Peak Performance Month: {max_month.date()}")
print(f"Lowest Performance Month: {min_month.date()}")

# 5. HISTORICAL TREND DETECTION
# Comparing first and last records to identify overall trajectory
if monthly_sales['Sales'].iloc[-1] > monthly_sales['Sales'].iloc[0]:
    print("Market Trend: 📈 Upward trajectory detected")
else:
    print("Market Trend: 📉 Downward trajectory detected")

# 6. PREDICTIVE MODELING (Holt-Winters Method)
# Implementing Triple Exponential Smoothing to account for Trend and Seasonality
model = ExponentialSmoothing(
    monthly_sales['Sales'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
)

fit = model.fit()

# 7. 12-MONTH FORECAST GENERATION
# Predicting sales volume for the upcoming 12 months
forecast = fit.forecast(12)

print("\n📈 PROJECTED SALES VALUES (Next 12 Months):")
print(forecast)

# Ensure the Output directory exists before saving
if not os.path.exists('../04_Output'):
    os.makedirs('../04_Output')

# Exporting forecast results to CSV for stakeholder reporting
forecast_df = forecast.reset_index()
forecast_df.columns = ['Forecast_Month', 'Projected_Sales']
forecast_df.to_csv('../04_Output/forecast_values.csv', index=False)

# 8. VISUALIZATION OF ACTUAL VS. PREDICTED SALES
plt.figure(figsize=(12, 6))

plt.plot(monthly_sales, label='Historical Sales Data', linewidth=2, color='#1f77b4')
plt.plot(forecast, label='AI-Powered Forecast (Next 12 Months)', linestyle='--', linewidth=2, color='#ff7f0e')

plt.title('Predictive Sales Analysis: Historical Performance vs 12-Month Forecast')
plt.xlabel('Timeline')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 9. EXPORTING VISUAL ASSETS
# Ensure the Screenshots directory exists
if not os.path.exists('../03_Screenshots'):
    os.makedirs('../03_Screenshots')

plt.savefig('../03_Screenshots/sales_forecast_plot.png', dpi=300)
plt.show()

print("\n✅ Predictive Analysis Complete!")
print("📁 Assets exported to: 03_Screenshots and 04_Output")