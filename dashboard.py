import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Perbandingan Model LSTM dan GRU", layout="wide")
st.title("📊 Perbandingan Model LSTM dan GRU dalam mempreidksi harga penutupan harian Bitcoin")

# =========================
# Fungsi bantu
# =========================
def load_and_prepare_csv(path):
    df = pd.read_csv(path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)
    return df

def load_eval(path):
    df = pd.read_csv(path)
    return df.iloc[0]  # Ambil baris pertama karena hanya satu baris

def plot_prediction(df, label):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["Actual"], label="Actual", color="blue")
    ax.plot(df.index, df["Predicted"], label="Predicted", color="orange", linestyle="--")
    ax.set_title(f"Prediksi Harga Bitcoin - {label}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (USD)")
    ax.grid(True)
    ax.legend()
    return fig

# =========================
# Pilihan periode
# =========================
periode = st.sidebar.selectbox("Pilih Periode", ["5 Tahun", "10 Tahun"])

if periode == "10 Tahun":
    lstm_file = "assets/csv_model_lstm_10tahun_bitcoin.csv"
    gru_file = "assets/csv_model_gru_10tahun_bitcoin.csv"
    eval_lstm_file = "assets/csv_evalscore_model_lstm_10tahun_bitcoin.csv"
    eval_gru_file = "assets/csv_evalscore_model_gru_10tahun_bitcoin.csv"
else:
    lstm_file = "assets/csv_model_lstm_5tahun_bitcoin.csv"
    gru_file = "assets/csv_model_gru_5tahun_bitcoin.csv"
    eval_lstm_file = "assets/csv_evalscore_model_lstm_5tahun_bitcoin.csv"
    eval_gru_file = "assets/csv_evalscore_model_gru_5tahun_bitcoin.csv"

# =========================
# Load Data
# =========================
df_lstm = load_and_prepare_csv(lstm_file)
df_gru = load_and_prepare_csv(gru_file)

eval_lstm = load_eval(eval_lstm_file)
eval_gru = load_eval(eval_gru_file)

# =========================
# Filter Tanggal
# =========================
min_date = df_lstm.index.min().date()
max_date = df_lstm.index.max().date()

st.sidebar.markdown("### Filter Tanggal Prediksi")
start_date = st.sidebar.date_input("Tanggal Mulai", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("Tanggal Akhir", min_value=start_date, max_value=max_date, value=max_date)

df_lstm_filtered = df_lstm.loc[str(start_date):str(end_date)]
df_gru_filtered = df_gru.loc[str(start_date):str(end_date)]

# =========================
# Layout
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🔵 LSTM - {periode}")
    st.pyplot(plot_prediction(df_lstm_filtered, f"LSTM - {periode}"))
    st.dataframe(df_lstm_filtered)

    st.markdown("**📈 Hasil Evaluasi LSTM:**")
    st.markdown(f"- RMSE: `{eval_lstm['RMSE']}`")
    st.markdown(f"- MAE: `{eval_lstm['MAE']}`")
    st.markdown(f"- R²: `{eval_lstm['R2']}`")
    st.markdown(f"- MAPE: `{eval_lstm['MAPE']}`")

with col2:
    st.markdown(f"### 🟠 GRU - {periode}")
    st.pyplot(plot_prediction(df_gru_filtered, f"GRU - {periode}"))
    st.dataframe(df_gru_filtered)

    st.markdown("**📈 Hasil Evaluasi GRU:**")
    st.markdown(f"- RMSE: `{eval_gru['RMSE']}`")
    st.markdown(f"- MAE: `{eval_gru['MAE']}`")
    st.markdown(f"- R²: `{eval_gru['R2']}`")
    st.markdown(f"- MAPE: `{eval_gru['MAPE']}`")