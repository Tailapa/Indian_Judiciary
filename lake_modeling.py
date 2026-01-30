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

def run_analysis_pipeline():
    if not os.path.exists('Bengaluru_Lakes_Time_Series_2010_2025.csv'):
        print("❌ Data file missing.")
        return

    df = pd.read_csv('Bengaluru_Lakes_Time_Series_2010_2025.csv')

    # 1. Time Series Plot
    plt.figure(figsize=(14, 8))
    # Plotting top 5 lakes by initial area for clarity if many are detected
    top_lakes = df[df['Year'] == 2010].nlargest(5, 'Lake_Area_sqm')['Lake_Name'].tolist()
    plot_df = df[df['Lake_Name'].isin(top_lakes)]

    sns.lineplot(data=plot_df, x='Year', y='Lake_Area_sqm', hue='Lake_Name', marker='o')
    plt.title('Bengaluru Lakes Area Trends (Top 5 Detected Lakes, 2010 - 2025)')
    plt.ylabel('Area (sqm)')
    plt.xlabel('Year')
    plt.legend(title='Lake Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('lake_area_trends.png')
    plt.close()
    print("📈 Saved: lake_area_trends.png")

    # 2. Regression Analysis
    X = df[['Encroachment_sqm']]
    y = df['Flood_Area_sqm']
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))

    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Encroachment_sqm', y='Flood_Area_sqm', scatter_kws={'alpha':0.3})
    plt.title(f'Regression: Encroachment vs Flooding (R² = {r2:.3f})')
    plt.xlabel('Encroachment Area (sqm)')
    plt.ylabel('Flood Area (sqm)')
    plt.tight_layout()
    plt.savefig('encroachment_vs_flooding_regression.png')
    plt.close()
    print(f"📉 Saved: encroachment_vs_flooding_regression.png (Impact Coefficient: {reg.coef_[0]:.4f})")

    # 3. Random Forest Model
    features = ['Year', 'Rainfall_mm', 'Encroachment_sqm', 'Green_Cover_sqm']
    X_m = df[features]
    y_m = df['Flood_Area_sqm']
    X_train, X_test, y_train, y_test = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"\n🤖 Flood Prediction Model (Random Forest):")
    print(f"   R² Score: {r2_score(y_test, y_pred):.3f}")

    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Flood Prediction')
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('flood_feature_importance.png')
    plt.close()
    print("📊 Saved: flood_feature_importance.png")

    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title('Correlation Matrix of Detected Lake Metrics')
    plt.tight_layout()
    plt.savefig('metrics_correlation_heatmap.png')
    plt.close()
    print("🔥 Saved: metrics_correlation_heatmap.png")

    # 5. Summary Report
    with open('Analysis_Summary.txt', 'w') as f:
        f.write("BENGALURU LAKES ANALYSIS SUMMARY (2010-2025)\n")
        f.write("============================================\n\n")
        f.write(f"Total Lakes Analyzed: {df['Lake_Name'].nunique()}\n")
        f.write(f"Encroachment-Flooding Regression R²: {r2:.3f}\n")
        f.write(f"Encroachment-Flooding Impact Coef: {reg.coef_[0]:.4f}\n")
        f.write("\nPredictive Model (Random Forest) Feature Importances:\n")
        for i in reversed(indices):
            f.write(f"- {features[i]}: {importances[i]:.4f}\n")
    print("📝 Saved: Analysis_Summary.txt")

if __name__ == "__main__":
    run_analysis_pipeline()
