import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import re


# ==== CONFIG ====
st.set_page_config(layout="wide", page_title="Bengaluru Skies")
# oa_token = st.secrets["OA_API_KEY"]
load_dotenv()
oa_token = os.getenv("OA_API_KEY")


# ==== FUNCTIONS ====
def get_latest_prediction_file():
    files = [f for f in os.listdir("predictions") if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join("predictions", files[0]) if files else None


def load_weather_data():
    file_path = get_latest_prediction_file()
    if not file_path:
        return None, None
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df, file_path


def get_trend_icon(curr, prev):
    if prev is None:
        return "⏺️"
    if curr > prev:
        return "🔺"
    elif curr < prev:
        return "🔻"
    else:
        return "⏺️"


# ==== SIDEBAR ====
st.sidebar.title("🔍 Navigate")
tab = st.sidebar.radio("Select a view:", ["🌤️ Today & Forecast", "💬 Ask AI"])


# ==== MAIN APP ====

# ==== MAIN APP ====
if "df" not in st.session_state or st.button("🔄 Refresh Data"):
    df, latest_file = load_weather_data()
    if df is None:
        st.error("No prediction file found.")
        st.stop()
    st.session_state.df = df
    st.session_state.latest_file = latest_file
else:
    df = st.session_state.df
    latest_file = st.session_state.latest_file


today = datetime.today().date()
regions = df["Region"].unique()


# ==== Format floats to 2 decimals ====
def format_response_numbers(response: str, decimals=2):
    pattern = rf"(-?\d+\.\d{{{decimals}}})\d*"
    return re.sub(pattern, r"\1", response)

# ==== TAB 1: TODAY & FORECAST ====
if tab == "🌤️ Today & Forecast":
    st.title("🌤️ Bengaluru Skies")
    # st.markdown(f"**Data file:** `{os.path.basename(latest_file)}`")

    st.subheader(f"🌞 Today's Weather — {today.strftime('%b %d, %Y')}")
    # st.write(df.head()) # remove

    # Header with info tooltip

    today_data = df[df["datetime"].dt.date == today]
    cols = st.columns(len(today_data))
    for i, (_, row) in enumerate(today_data.iterrows()):
        with cols[i]:
            st.metric(
                label=f"📍 {row['Region']}",
                value=f"{row['temp']:.1f}°C"
                # help="Average temperature for the day"
            )
            st.write(f"🔆 Solar energy: {row['solarenergy']}MJ/m²")
            st.write(f"🌬️ Windgust: {row['windgust']} km/h")
            st.write(f"Conditions: {row['conditions']}")

    # Forecast cards for next 5 days
    st.markdown("---")

    st.subheader("📆 7-Day Weather (Yesterday + Today + Next 5 Days)")
    start_date = today - timedelta(days=1)
    end_date = today + timedelta(days=5)
    forecast_df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)].copy()
    forecast_df["temperature"] = forecast_df.apply(
        lambda x: x["predicted_temp"] if not pd.isna(x["predicted_temp"]) else x["temp"], axis=1
    )

    for region in regions:
        st.markdown(f"### 📍 {region}")
        # region_df = forecast_df[forecast_df["Region"] == region].set_index(forecast_df["datetime"].dt.date)
        region_df = forecast_df[forecast_df["Region"] == region].copy()
        region_df["date"] = region_df["datetime"].dt.date
        region_df = region_df.set_index("date")

        dates = pd.date_range(start_date, end_date).date
        cols = st.columns(len(dates))
        prev_temp = None
        for i, date in enumerate(dates):
            with cols[i]:
                if date in region_df.index:
                    row = region_df.loc[date]
                    temp = round(row["temperature"], 1)
                    trend = get_trend_icon(temp, prev_temp)
                    label = "Today" if date == today else date.strftime("%a %d")
                    # st.metric(
                    #     label=f"{label} {trend}",
                    #     value=f"{temp}°C"
                    #     # help="Average temperature for the day"
                    # )
                    # st.caption(f"💧 {row['humidity']}% | 🌬️ {row['windgust']} km/h")
                    # Set background color based on the day
                    if date == today:
                        bg_color = "#4CAF50"  # slightly deeper green for today
                    elif date == today - timedelta(days=1):
                        bg_color = "#81C784"  # light green for yesterday
                    else:
                        bg_color = "#9575CD"  # light purple for future days

                    # Build card with HTML
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {bg_color};
                            padding: 1rem;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.05);
                        ">
                            <strong>{label} {trend}</strong><br>
                            <span style="font-size: 1.5em;">{temp}°C</span><br>
                            
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    prev_temp = temp
                else:
                    st.metric(label=date.strftime("%a %d"), value="—")




elif tab == "💬 Ask AI":
    st.title("💬 Ask Weather AI")
    st.markdown(f"You're chatting with a weather assistant using a csv file with weather data for Whitefield, Hebbal, Devanahalli, Electronic City, Mysore Road, BTM Layout. The data range is from 2023 to present day.")

    # query = st.text_input("Type your weather question:")
    query = st.text_input(
        "Type a simple weather question:",
        placeholder="E.g., When and where was the hottest day in May 2025? Or, Average temp during December 2023 in hebbal?"
    )

    if query:
        with st.spinner("Thinking..."):
            try:
                llm = OpenAI(temperature=0, max_tokens=400, api_key=oa_token)
                # agent = create_csv_agent(llm, latest_file, verbose=True, allow_dangerous_code=True)
                agent = create_csv_agent(
                        llm,
                        latest_file,
                        verbose=False,
                        allow_dangerous_code=True,
                        agent_type="zero-shot-react-description",
                        max_iterations=10,        # default is 5
                        max_execution_time=120,   # in seconds
                    )
                response = agent.run(query)
                formatted = format_response_numbers(response)
                st.success(formatted)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

