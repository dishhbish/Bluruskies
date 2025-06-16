import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🌤️ Bangalore Weather Forecast Dashboard")

# === Refresh Button ===
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()


# === Get latest prediction file ===
def get_latest_prediction_file():
    files = [f for f in os.listdir("predictions") if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join("predictions", files[0]) if files else None


file_path = get_latest_prediction_file()
if not file_path:
    st.error("No prediction file found.")
    st.stop()

# === Load data ===
df = pd.read_csv(file_path, parse_dates=["datetime"])
df["datetime"] = pd.to_datetime(df["datetime"])

# === Filter for today ===
today = datetime.today().date()
today_data = df[df["datetime"].dt.date == today]
st.subheader(f"🌞 Today's Weather - {today.strftime('%b %d, %Y')}")

# === Display metrics as cards ===
cols = st.columns(len(today_data))
for i, (_, row) in enumerate(today_data.iterrows()):
    with cols[i]:
        st.metric(label=f"📍 {row['Region']}", value=f"{row['temp']:.1f}°C", help="Actual temperature")
        st.write(f"💧 Humidity: {row['humidity']}%")
        st.write(f"🌫️ Windgust: {row['windgust']}km/h")

# === Line plot: Past 3 + Next 5 days ===


# Define range
# start_date = today - timedelta(days=3)
# end_date = today + timedelta(days=5)
# plot_df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)].copy()
#
# # Use predicted_temp where available, else temp
# plot_df["temperature"] = plot_df.apply(
#     lambda x: x["predicted_temp"] if not pd.isna(x["predicted_temp"]) else x["temp"], axis=1
# )

# # Plot
# fig, ax = plt.subplots(figsize=(10, 5))
# regions = plot_df["Region"].unique()
#
# for region in regions:
#     region_df = plot_df[plot_df["Region"] == region].sort_values("datetime")
#     ax.plot(region_df["datetime"], region_df["temperature"], label=region)
#
#     # Add marker at the transition point (today)
#     transition_point = region_df[region_df["datetime"].dt.date == today]
#     if not transition_point.empty:
#         ax.scatter(
#             transition_point["datetime"],
#             transition_point["temperature"],
#             color=ax.get_lines()[-1].get_color(),
#             edgecolor="black",
#             zorder=5
#             # ,label=f"{region} (today)"
#         )
#
# ax.set_title("Region-wise Temperature Trend (Past 3 + Predicted for next 5 Days)")
# ax.set_xlabel("Date")
# ax.set_ylabel("Temperature (°C)")
# plt.xticks(rotation=30)
# ax.legend()
# st.pyplot(fig)

# === Enhanced Line plot: Past 3 + Next 5 days ===
from matplotlib.dates import DateFormatter

# === Enhanced Line plot: Past 3 + Next 5 days ===
st.subheader("📈 Forecast: Past 3 and Next 5 Days (All Regions)")

# Filter date range
start_date = today - timedelta(days=3)
end_date = today + timedelta(days=5)
plot_df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)].copy()

# Choose best temperature value
plot_df["temperature"] = plot_df.apply(
    lambda x: x["predicted_temp"] if not pd.isna(x["predicted_temp"]) else x["temp"], axis=1
)

# Plot setup
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
palette = sns.color_palette("Set2", n_colors=len(plot_df["Region"].unique()))
regions = plot_df["Region"].unique()

for i, region in enumerate(regions):
    region_df = plot_df[plot_df["Region"] == region].sort_values("datetime")

    ax.plot(
        region_df["datetime"],
        region_df["temperature"],
        label=region,
        color=palette[i],
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.5,
    )

    # Mark today's point
    today_point = region_df[region_df["datetime"].dt.date == today]
    if not today_point.empty:
        ax.scatter(
            today_point["datetime"],
            today_point["temperature"],
            color=palette[i],
            edgecolor="black",
            zorder=5,
            s=100,
            linewidth=1.5
        )

# Labels and styling
ax.set_title("🌡️ Region-wise Temperature Trend (Past 3 + Next 5 Days)", fontsize=16, weight='bold')
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.tick_params(axis='both', labelsize=10)

# Set x-axis date format to dd/mm/yyyy and keep horizontal
ax.xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Move legend outside
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

st.pyplot(fig)
