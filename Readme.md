# 🌤️ Bengaluru Skies: Weather Forecasting & Insight Platform using Machine Learning

**Bengaluru Skies** is an end-to-end ML-powered weather analytics platform that:

- **Fetches historical & daily weather data** for key Bengaluru localities using the [Visual Crossing API].
- **Trains time-series models (XGBoost)** to forecast temperatures for the next 7 days, using engineered temporal and meteorological features.
- **Implements RandomizedSearchCV** with time-series split to optimize hyperparameters and improve prediction accuracy.
- **Visualizes predictions** and historical trends interactively with **Streamlit**, providing a sleek UI for daily updates and 7-day forecasts.
- **Includes a natural language chatbot** powered by LangChain + OpenAI, enabling CSV-based Q&A on weather trends using LLMs.

All forecasts and updates are saved and versioned automatically, making this a scalable and extensible ML deployment pipeline.
