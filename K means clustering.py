import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset
np.random.seed(42)
years = np.arange(2000, 2025)
months = np.tile(np.arange(1, 13), len(years))
year_repeated = np.repeat(years, 12)

# Simulate average temperatures with seasonality
base_temperatures = np.sin(2 * np.pi * months / 12) * 10 + 15  # Seasonal effect
random_noise = np.random.normal(0, 2, len(base_temperatures))  # Random fluctuations
avg_temperatures = base_temperatures + random_noise

# Create DataFrame
df = pd.DataFrame({
    'Year': year_repeated,
    'Month': months,
    'AvgTemperature': avg_temperatures
})

# Create a Date column
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

# Sort by date
df = df.sort_values('Date')

# Set Date as index before resampling
df.set_index('Date', inplace=True)

# Aggregate data to monthly level
df = df.resample('ME').mean()

# Extract Month names for plotting
df['Month'] = df.index.month_name()

# Plot original time series
plt.figure(figsize=(12, 5))
plt.plot(df['Month'], df['AvgTemperature'], label='Original Data')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.title('Time Series Plot - Synthetic Data')
plt.legend()
plt.show()

# Ensure sufficient data for decomposition
if len(df) >= 24:
    decomposition = seasonal_decompose(df['AvgTemperature'], period=12, model='additive', extrapolate_trend='freq')

    # Plot decomposition
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    decomposition.trend.plot(ax=axes[0], title='Trend')
    decomposition.seasonal.plot(ax=axes[1], title='Seasonality')
    decomposition.resid.plot(ax=axes[2], title='Residuals')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough observations for seasonal decomposition.")

# Moving average smoothing
window_size = 3  # Modify as needed
df['Moving_Avg'] = df['AvgTemperature'].rolling(window=window_size).mean()

# Plot moving average
plt.figure(figsize=(12, 5))
plt.plot(df['Month'], df['AvgTemperature'], label='Original Data', alpha=0.5)
plt.plot(df['Month'], df['Moving_Avg'], color='red', label=f'{window_size}-Month Moving Average')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.title('Moving Average Smoothing - Synthetic Data')
plt.legend()
plt.show()

# K-Means Clustering Analysis
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['AvgTemperature']].dropna())

# Determine optimal clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Apply K-Means Clustering with optimal clusters (assuming 3 based on the elbow method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Plot Clusters
plt.figure(figsize=(12, 5))
plt.scatter(df['Month'], df['AvgTemperature'], c=df['Cluster'], cmap='viridis', label='Clustered Data')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.title('K-Means Clustering - Temperature Patterns')
plt.legend()
plt.show()
