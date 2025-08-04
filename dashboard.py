import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Perbandingan Model LSTM dan GRU", layout="wide")
st.title("📊 Perbandingan Model LSTM dan GRU Dalam Memprediksi Harga Penutupan Harian Bitcoin")

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
    ax.plot(df.index, df["Predicted"], label="Predicted", color="orange", linestyle="--", linewidth=2)
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
def render_eval_table(title, eval_dict):
    st.markdown(f"**📈 {title}**")

    html = f"""
    <style>
        .styled-table {{
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 16px;
            width: 100%;
            text-align: center;
        }}
        .styled-table thead tr {{
            background-color: #2c2f33;
            color: #ffffff;
        }}
        .styled-table td, .styled-table th {{
            border: 1px solid #dddddd;
            padding: 8px;
        }}
        .styled-table tbody td:nth-child(2) {{
            color: #00ff88;
            font-weight: bold;
        }}
    </style>
    <table class="styled-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Hasil</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>RMSE</td><td>{float(eval_dict['RMSE']):,.2f}</td></tr>
            <tr><td>MAE</td><td>{float(eval_dict['MAE']):,.2f}</td></tr>
            <tr><td>R²</td><td>{float(eval_dict['R2']):.4f}</td></tr>
            <tr><td>MAPE</td><td>{eval_dict['MAPE']}</td></tr>
        </tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Layout
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🔵 LSTM - {periode}")
    st.pyplot(plot_prediction(df_lstm_filtered, f"LSTM - {periode}"))
    st.dataframe(df_lstm_filtered)
    render_eval_table("Hasil Evaluasi LSTM", eval_lstm)

with col2:
    st.markdown(f"### 🟠 GRU - {periode}")
    st.pyplot(plot_prediction(df_gru_filtered, f"GRU - {periode}"))
    st.dataframe(df_gru_filtered)
    render_eval_table("Hasil Evaluasi GRU", eval_gru)
