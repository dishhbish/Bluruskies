# Bengaluru Skies – ML Weather Forecasting & Interactive LLM Assistant

**Bengaluru Skies** is an end-to-end machine learning pipeline and Streamlit-based weather intelligence platform that delivers daily forecasts and LLM-powered insights for key regions in Bengaluru, India.

---

## Features

### Weather Data Pipeline
- Uses **Visual Crossing API** to continuously fetch and update localized weather datasets (temperature, humidity, dew point, conditions) for 8 Bengaluru regions.
- Automatically persists updates in time-indexed `.csv` files for reproducible training and inference.

### Forecasting Engine
- Implements **XGBoost Regressor** with:
  - **Lag-based temporal features** and holiday/weekend flags.
  - **Categorical encoding** of weather conditions via one-hot encoding.
  - **TimeSeriesSplit CV** with **RandomizedSearchCV** for optimal hyperparameter tuning.
- Trains on 80% of historical data and evaluates RMSE on the holdout split.
- Predicts 7-day rolling forecasts using recent lags, seasonal averages, and calendar-based signals.

### Frontend Dashboard (Streamlit)
- **Region-wise visualization** of today’s weather and 7-day outlook with temperature trend icons.
- Stylized cards with temperature, wind, and solar energy indicators.
- Includes refresh capability and responsive layout for seamless exploration.

### Chatbot Integration
- Interactive **natural language weather assistant** powered by:
  - LangChain CSV agent + OpenAI GPT models.
  - Supports queries like:
    - _"Where was the hottest day in May 2025?"_
    - _"What was the average temperature in Hebbal during December 2023?"_

---

## Tech Stack

| Area | Tools/Technologies |
|------|--------------------|
| Data Fetching | `requests`, `Visual Crossing API`, `pandas` |
| ML Forecasting | `xgboost`, `sklearn` (`RandomizedSearchCV`, `TimeSeriesSplit`) |
| Feature Engineering | Lag variables, one-hot encoding, calendar/holiday flags |
| Visualization | `streamlit`, `matplotlib` |
| NLP Integration | `LangChain`, `OpenAI` |


---


