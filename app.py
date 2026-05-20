import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import io
import joblib
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main { background: #0a0e1a; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    color: #64748b;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.normal-val { color: #34d399; }
.ddos-val   { color: #f87171; }
.port-val   { color: #fb923c; }
.total-val  { color: #38bdf8; }

.anomaly-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-normal  { background: #064e3b; color: #34d399; }
.badge-ddos    { background: #7f1d1d; color: #fca5a5; }
.badge-portscan{ background: #7c2d12; color: #fdba74; }

.stDataFrame { border-radius: 10px; }

div[data-testid="stFileUploader"] {
    background: rgba(56,189,248,0.05);
    border: 2px dashed rgba(56,189,248,0.3);
    border-radius: 14px;
    padding: 1rem;
}

div[data-testid="stSidebar"] {
    background: #080c18;
    border-right: 1px solid rgba(255,255,255,0.06);
}

h2, h3 { font-family: 'Syne', sans-serif; }

.status-bar {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #38bdf8;
    background: rgba(56,189,248,0.08);
    border-left: 3px solid #38bdf8;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: generate synthetic IoT training data ──────────────────────────────
@st.cache_resource
def train_model():
    """Train RF on synthetic IoT traffic data calibrated for IoT characteristics."""
    np.random.seed(42)
    n = 6000

    # Normal IoT traffic: small packets, short flows, low rates
    normal = pd.DataFrame({
        'Flow_Duration': np.random.exponential(0.3, n // 3),
        'Packets':       np.random.randint(2, 80, n // 3),
        'Bytes':         np.random.randint(40, 1500, n // 3),
        'Protocol':      np.random.choice([6, 17], n // 3, p=[0.6, 0.4]),  # TCP/UDP
        'Label': 'Normal'
    })

    # DDoS: massive packet/byte volume, many flows
    ddos = pd.DataFrame({
        'Flow_Duration': np.random.uniform(0.5, 30.0, n // 3),
        'Packets':       np.random.randint(5000, 50000, n // 3),
        'Bytes':         np.random.randint(100000, 2000000, n // 3),
        'Protocol':      np.random.choice([6, 17, 1], n // 3, p=[0.5, 0.3, 0.2]),
        'Label': 'DDoS'
    })

    # PortScan: many tiny flows, UDP/TCP mix, very few bytes
    portscan = pd.DataFrame({
        'Flow_Duration': np.random.uniform(0.001, 0.2, n // 3),
        'Packets':       np.random.randint(1, 4, n // 3),
        'Bytes':         np.random.randint(40, 120, n // 3),
        'Protocol':      np.random.choice([6, 17], n // 3, p=[0.8, 0.2]),
        'Label': 'PortScan'
    })

    data = pd.concat([normal, ddos, portscan], ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    le = LabelEncoder()
    data['Label_enc'] = le.fit_transform(data['Label'])

    # Engineered features (IoT-specific)
    eps = 1e-9
    data['Packet_Rate']    = data['Packets'] / (data['Flow_Duration'] + eps)
    data['Byte_Rate']      = data['Bytes']   / (data['Flow_Duration'] + eps)
    data['Packet_Size_Avg']= data['Bytes']   / (data['Packets'] + eps)

    feature_cols = ['Flow_Duration','Packets','Bytes','Protocol',
                    'Packet_Rate','Byte_Rate','Packet_Size_Avg']

    X = data[feature_cols]
    y = data['Label_enc']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # IoT-tuned RF parameters
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y_res)

    return rf, scaler, le, feature_cols


# ── Helper: preprocess uploaded dataframe ────────────────────────────────────
def preprocess(df, scaler, feature_cols):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    eps = 1e-9
    if 'Packet_Rate' not in df.columns:
        df['Packet_Rate']     = df['Packets'] / (df['Flow_Duration'] + eps)
    if 'Byte_Rate' not in df.columns:
        df['Byte_Rate']       = df['Bytes']   / (df['Flow_Duration'] + eps)
    if 'Packet_Size_Avg' not in df.columns:
        df['Packet_Size_Avg'] = df['Bytes']   / (df['Packets'] + eps)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"В датасете отсутствуют колонки: {missing}")
        return None, None

    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    return df, X_scaled


# ── Helper: load CSV from ZIP ─────────────────────────────────────────────────
def load_csv_from_zip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            return None, "В архиве не найдено CSV-файлов."
        # pick first csv
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)
        return df, csv_files[0]


# ── LABEL COLORS ──────────────────────────────────────────────────────────────
LABEL_COLORS = {
    'Normal':   '#34d399',
    'DDoS':     '#f87171',
    'PortScan': '#fb923c',
}

BADGE_CLASS = {
    'Normal':   'badge-normal',
    'DDoS':     'badge-ddos',
    'PortScan': 'badge-portscan',
}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">🛡️ IoT Anomaly Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ОБНАРУЖЕНИЕ АНОМАЛИЙ В СЕТЕВОМ ТРАФИКЕ IoT-УСТРОЙСТВ</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Параметры")
    confidence_threshold = st.slider(
        "Порог уверенности (%)", 50, 99, 70,
        help="Минимальная уверенность модели для пометки аномалии"
    )
    show_probabilities = st.checkbox("Показывать вероятности", value=True)
    show_feature_importance = st.checkbox("График важности признаков", value=True)

    st.markdown("---")
    st.markdown("**Типы аномалий:**")
    st.markdown("🔴 **DDoS** — распределённая атака отказа в обслуживании")
    st.markdown("🟠 **PortScan** — сканирование портов")
    st.markdown("🟢 **Normal** — нормальный трафик")

    st.markdown("---")
    st.markdown("**Ожидаемые колонки CSV:**")
    st.code("Flow_Duration\nPackets\nBytes\nProtocol", language="text")

# ── Train model (cached) ──────────────────────────────────────────────────────
with st.spinner("Загрузка модели..."):
    rf_model, scaler, le, feature_cols = train_model()

st.markdown('<div class="status-bar">✅ Модель загружена и готова к анализу</div>', unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
st.markdown("## 📂 Загрузите датасет")
uploaded = st.file_uploader(
    "Перетащите ZIP-архив с CSV-файлом сюда",
    type=["zip", "csv"],
    help="ZIP-архив должен содержать хотя бы один CSV-файл"
)

if uploaded is not None:
    # Load data
    if uploaded.name.endswith('.zip'):
        df_raw, fname = load_csv_from_zip(uploaded.read())
        if df_raw is None:
            st.error(fname)
            st.stop()
        st.markdown(f'<div class="status-bar">📄 Файл из архива: <b>{fname}</b> — {len(df_raw):,} строк</div>',
                    unsafe_allow_html=True)
    else:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f'<div class="status-bar">📄 Файл: <b>{uploaded.name}</b> — {len(df_raw):,} строк</div>',
                    unsafe_allow_html=True)

    # Preprocess & predict
    df_proc, X_scaled = preprocess(df_raw.copy(), scaler, feature_cols)

    if X_scaled is not None:
        proba = rf_model.predict_proba(X_scaled)
        preds = rf_model.predict(X_scaled)
        pred_labels = le.inverse_transform(preds)
        max_conf = proba.max(axis=1) * 100

        df_proc = df_proc.reset_index(drop=True)
        df_proc['Предсказание'] = pred_labels
        df_proc['Уверенность (%)'] = max_conf.round(1)

        # Apply confidence threshold
        df_proc['Статус'] = np.where(
            max_conf >= confidence_threshold,
            df_proc['Предсказание'],
            'Неопределено'
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        st.markdown("## 📊 Результаты анализа")

        total   = len(df_proc)
        n_norm  = (df_proc['Статус'] == 'Normal').sum()
        n_ddos  = (df_proc['Статус'] == 'DDoS').sum()
        n_port  = (df_proc['Статус'] == 'PortScan').sum()
        n_anom  = n_ddos + n_port

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-val total-val">{total:,}</div>
                <div class="metric-label">Всего записей</div></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-val normal-val">{n_norm:,}</div>
                <div class="metric-label">Нормальный трафик</div></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-val ddos-val">{n_ddos:,}</div>
                <div class="metric-label">DDoS атак</div></div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-val port-val">{n_port:,}</div>
                <div class="metric-label">PortScan атак</div></div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ───────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Распределение классов")
            counts = df_proc['Статус'].value_counts()
            colors_pie = [LABEL_COLORS.get(l, '#94a3b8') for l in counts.index]
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
            ax.pie(counts.values, labels=counts.index, colors=colors_pie,
                   autopct='%1.1f%%', startangle=90,
                   textprops={'color': '#e2e8f0', 'fontsize': 11})
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("### Уверенность модели")
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='none')
            for label, color in LABEL_COLORS.items():
                subset = df_proc[df_proc['Статус'] == label]['Уверенность (%)']
                if len(subset) > 0:
                    ax2.hist(subset, bins=20, alpha=0.7, label=label, color=color)
            ax2.set_xlabel('Уверенность (%)', color='#94a3b8')
            ax2.set_ylabel('Количество', color='#94a3b8')
            ax2.tick_params(colors='#94a3b8')
            ax2.set_facecolor('#0f1629')
            ax2.spines['bottom'].set_color('#334155')
            ax2.spines['left'].set_color('#334155')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(facecolor='#0a0e1a', labelcolor='#e2e8f0')
            fig2.patch.set_alpha(0)
            st.pyplot(fig2)
            plt.close()

        # ── Feature importance ────────────────────────────────────────────────
        if show_feature_importance:
            st.markdown("### 🔍 Важность признаков (Random Forest)")
            importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
            importances = importances.sort_values()
            fig3, ax3 = plt.subplots(figsize=(8, 3.5), facecolor='none')
            colors_bar = ['#38bdf8' if v > importances.median() else '#334155'
                          for v in importances.values]
            importances.plot(kind='barh', ax=ax3, color=colors_bar)
            ax3.set_facecolor('#0f1629')
            ax3.tick_params(colors='#94a3b8')
            ax3.spines['bottom'].set_color('#334155')
            ax3.spines['left'].set_color('#334155')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            fig3.patch.set_alpha(0)
            st.pyplot(fig3)
            plt.close()

        # ── Anomaly table ─────────────────────────────────────────────────────
        st.markdown("## 🚨 Обнаруженные аномалии")

        anomalies = df_proc[df_proc['Статус'].isin(['DDoS', 'PortScan'])].copy()

        if len(anomalies) == 0:
            st.success("✅ Аномалий не обнаружено!")
        else:
            st.error(f"⚠️ Обнаружено **{len(anomalies):,}** аномальных записей из {total:,}")

            # Show probabilities if enabled
            if show_probabilities:
                classes = le.classes_
                for i, cls in enumerate(classes):
                    anomalies[f'P({cls}) %'] = (proba[anomalies.index, i] * 100).round(1)

            # Highlight table
            display_cols = feature_cols + ['Статус', 'Уверенность (%)']
            if show_probabilities:
                display_cols += [f'P({c}) %' for c in le.classes_]

            def color_row(row):
                color_map = {'DDoS': '#3d0000', 'PortScan': '#3d1a00'}
                bg = color_map.get(row['Статус'], '')
                return [f'background-color: {bg}'] * len(row)

            styled = (anomalies[display_cols]
                      .reset_index(drop=True)
                      .style.apply(color_row, axis=1)
                      .format({'Уверенность (%)': '{:.1f}%'}))
            st.dataframe(styled, use_container_width=True, height=400)

            # Download
            csv_out = anomalies[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Скачать аномалии (CSV)",
                data=csv_out,
                file_name="anomalies_detected.csv",
                mime="text/csv"
            )

        # ── Full table ────────────────────────────────────────────────────────
        with st.expander("📋 Показать полную таблицу"):
            cols_show = feature_cols + ['Статус', 'Уверенность (%)']
            st.dataframe(df_proc[cols_show].reset_index(drop=True),
                         use_container_width=True, height=350)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📡</div>
        <div style="font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #38bdf8;">
            Загрузите ZIP-архив с CSV-файлом для начала анализа
        </div>
        <div style="margin-top: 1rem; font-size: 0.85rem;">
            Поддерживаемые форматы: .zip (с CSV внутри) или .csv напрямую
        </div>
    </div>
    """, unsafe_allow_html=True)
