import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load Data
df = pd.read_csv('Bengaluru_Lakes_Time_Series_2010_2025.csv')

# 2. Time Series Visualizations
def plot_time_series(df):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='Year', y='Lake_Area_sqm', hue='Lake_Name', marker='o')
    plt.title('Bengaluru Lakes Area Trends (2010 - 2025)')
    plt.ylabel('Area (sqm)')
    plt.xlabel('Year')
    plt.legend(title='Lake Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('lake_area_trends.png')
    plt.close()
    print("📈 Saved: lake_area_trends.png")

# 3. Regression Analysis: Encroachment vs Flooding
def perform_regression(df):
    X = df[['Encroachment_sqm']]
    y = df['Flood_Area_sqm']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    coef = model.coef_[0]

    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Encroachment_sqm', y='Flood_Area_sqm', scatter_kws={'alpha':0.5})
    plt.title(f'Regression: Encroachment vs Flooding (R² = {r2:.3f})')
    plt.xlabel('Encroachment Area (sqm)')
    plt.ylabel('Flood Area (sqm)')
    plt.tight_layout()
    plt.savefig('encroachment_vs_flooding_regression.png')
    plt.close()
    print(f"📉 Saved: encroachment_vs_flooding_regression.png (Coef: {coef:.4f})")

    return model

# 4. Predictive Flood Model (Random Forest)
def build_flood_model(df):
    # Features: Year, Rainfall, Encroachment, Green_Cover
    features = ['Year', 'Rainfall_mm', 'Encroachment_sqm', 'Green_Cover_sqm']
    X = df[features]
    y = df['Flood_Area_sqm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n🤖 Flood Prediction Model (Random Forest):")
    print(f"   Mean Squared Error: {mse:.2f}")
    print(f"   R² Score: {r2:.3f}")

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Flood Prediction')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('flood_feature_importance.png')
    plt.close()
    print("📊 Saved: flood_feature_importance.png")

    return model

# 5. Correlation Heatmap
def plot_correlation(df):
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Lake Metrics')
    plt.tight_layout()
    plt.savefig('metrics_correlation_heatmap.png')
    plt.close()
    print("🔥 Saved: metrics_correlation_heatmap.png")

if __name__ == "__main__":
    if os.path.exists('Bengaluru_Lakes_Time_Series_2010_2025.csv'):
        plot_time_series(df)
        perform_regression(df)
        build_flood_model(df)
        plot_correlation(df)
        print("\n✅ Analysis pipeline complete.")
    else:
        print("❌ Data file missing.")
