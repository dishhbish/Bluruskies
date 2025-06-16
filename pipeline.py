import os
import io
import requests
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta, datetime
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# === Configuration ===
API_KEY = os.getenv("vc_api_key") or "your_default_api_key"
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
DATA_DIR = "data"
PREDICTION_DIR = "predictions"

os.makedirs(PREDICTION_DIR, exist_ok=True)

REGIONS = {
    "whitefield": "12.9698,77.7499",
    "hebbal": "13.0358,77.5912",
    "electronic_city": "12.8452,77.6600",
    "indiranagar": "12.9719,77.6412",
    "koramangala": "12.9352,77.6147",
    "btm_layout": "12.9166,77.6101",
    "mysore_road": "12.9425,77.5252",
    "devanahalli": "13.2437,77.7139"
}

india_holidays = holidays.India()
today = datetime.today().date()
today_str = today.strftime('%Y%m%d')

# === Step 1: Update historical data ===
print("\n📥 Updating regional data from Visual Crossing API...")

for region_name, latlon in REGIONS.items():
    filename = os.path.join(DATA_DIR, f"{region_name}_2023_2025.csv")
    if not os.path.exists(filename):
        print(f"⚠️  File not found: {filename}. Skipping...")
        continue

    df = pd.read_csv(filename, parse_dates=["datetime"])
    latest_date = df["datetime"].max().date()

    if latest_date >= today:
        print(f"✅ {region_name}: Already up to date.")
        continue

    start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": API_KEY,
        "contentType": "csv"
    }

    url = f"{BASE_URL}/{latlon}/{start_date}/{end_date}"
    response = requests.get(url, params=params)

    if response.status_code == 200:
        new_data = pd.read_csv(io.StringIO(response.text), parse_dates=["datetime"])
        updated_df = pd.concat([df, new_data]).drop_duplicates(subset="datetime").sort_values("datetime")
        updated_df.to_csv(filename, index=False)
        print(f"✅ {region_name}: Updated data written to {filename}")
    else:
        print(f"❌ {region_name}: Error {response.status_code} - {response.text}")

# === Step 2: Forecast using XGBoost ===
print("\n🔮 Generating forecasts...")

def is_holiday(date):
    return int(date.weekday() >= 5 or date in india_holidays)

def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

all_forecasts = []

for region_key, latlon in REGIONS.items():
    file = os.path.join(DATA_DIR, f"{region_key}_2023_2025.csv")
    if not os.path.exists(file):
        print(f"⚠️ File missing for {region_key}. Skipping.")
        continue

    region_name = region_key.replace("_", " ").title()
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df['is_holiday'] = df.index.to_series().apply(is_holiday)

    df_region = df[["temp", "humidity", "dew", "conditions", "is_holiday"]]
    df_region = create_features(df_region)

    for lag in [1, 2, 3]:
        df_region[f'temp_lag_{lag}'] = df_region['temp'].shift(lag)

    df_region.dropna(inplace=True)
    df_region = pd.get_dummies(df_region, columns=['conditions'], prefix='cond')

    drop_cols = ['windspeed','precip','datetime','name','tempmax', 'tempmin', 'solarradiation', 'visibility', 'cloudcover', 'stations', 'Region','sealevelpressure']
    df_region.drop(columns=[col for col in drop_cols if col in df_region], inplace=True, errors='ignore')

    split = int(len(df_region) * 0.8)
    split_date = df_region.index[split]
    train = df_region[df_region.index < split_date]
    test = df_region[df_region.index >= split_date]

    TARGET = 'temp'
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = XGBRegressor(objective='reg:squarederror', early_stopping_rounds=50, verbosity=0)
    param_grid = {
        'n_estimators': [100, 250, 400],
        'max_depth': [1, 2, 3],
        'learning_rate': [0.01, 0.035, 0.06],
        'subsample': [0.6, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 3],
        'reg_alpha': [0, 0.1, 0.25],
        'reg_lambda': [0.75, 0.9, 1],
    }

    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=30, scoring='neg_root_mean_squared_error', cv=TimeSeriesSplit(n_splits=3), random_state=42, n_jobs=-1)
    search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    best_model = search.best_estimator_
    print(f"✅ {region_name}: RMSE = {np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))):.2f}")

    # Forecast next 7 days
    avg_humidity = df_region.groupby(df_region.index.dayofyear)['humidity'].mean()
    avg_dew = df_region.groupby(df_region.index.dayofyear)['dew'].mean()

    last_date = df_region.index.max()
    future_df = pd.DataFrame(index=pd.date_range(start=last_date + timedelta(days=1), periods=7))
    future_df = create_features(future_df)
    future_df['is_holiday'] = future_df.index.to_series().apply(is_holiday)
    future_df['humidity'] = future_df['dayofyear'].map(avg_humidity)
    future_df['dew'] = future_df['dayofyear'].map(avg_dew)

    for lag in [1, 2, 3]:
        future_df[f'temp_lag_{lag}'] = np.nan

    for i in range(len(future_df)):
        idx = future_df.index[i]
        if i == 0:
            future_df.loc[idx, 'temp_lag_1'] = df_region.iloc[-1]['temp']
            future_df.loc[idx, 'temp_lag_2'] = df_region.iloc[-2]['temp']
            future_df.loc[idx, 'temp_lag_3'] = df_region.iloc[-3]['temp']
        else:
            for lag in [1, 2, 3]:
                future_df.loc[idx, f'temp_lag_{lag}'] = future_df.iloc[i-lag]['temp_pred'] if i >= lag else df_region.iloc[-lag]['temp']

        # Add dummy columns missing from test set
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0

        row = future_df.loc[[idx], X_train.columns]
        future_df.loc[idx, 'temp_pred'] = best_model.predict(row)[0]

    future_df['Region'] = region_name
    all_forecasts.append(future_df[['temp_pred', 'Region']])

# === Step 3: Combine and Save ===
final_forecasts = pd.concat(all_forecasts)
final_forecasts.to_csv(os.path.join(PREDICTION_DIR, f'predictions_{today_str}.csv'))

# Merge with raw regional data
combined_df = []
for region_key in REGIONS.keys():
    file = os.path.join(DATA_DIR, f"{region_key}_2023_2025.csv")
    if not os.path.exists(file):
        continue
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Region'] = region_key.replace("_", " ").title()

    pred_df = final_forecasts[final_forecasts["Region"] == df["Region"].iloc[0]].copy()
    pred_df = pred_df.rename(columns={'temp_pred': 'predicted_temp'})
    pred_df = pred_df.reset_index().rename(columns={'index': 'datetime'})

    merged = pd.merge(df, pred_df, on=["datetime", "Region"], how="outer")
    combined_df.append(merged)

if combined_df:
    final_combined = pd.concat(combined_df).sort_values(["Region", "datetime"])
    final_combined.to_csv(os.path.join(PREDICTION_DIR, f"all_regions_with_predictions_{today_str}.csv"), index=False)
    print(f"✅ All region forecasts saved to predictions/all_regions_with_predictions_{today_str}.csv")
else:
    print("⚠️ No combined data saved. Check input files or forecast generation.")
