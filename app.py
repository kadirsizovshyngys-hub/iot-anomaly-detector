
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IoT Threat Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] { background:#040810; color:#c9d1d9; font-family: monospace; }
.stApp { background:#040810; }
.stApp::before {
    content:''; position:fixed; inset:0;
    background-image:linear-gradient(rgba(0,255,180,.04) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(0,255,180,.04) 1px,transparent 1px);
    background-size:40px 40px; pointer-events:none;
}
.hero { text-align:center; padding:2rem 1rem 1rem; }
.hero-title { font-size:2.6rem; color:#00ffb4; font-weight:900; letter-spacing:.08em; }
.hero-sub { color:#4c6f83; letter-spacing:.18em; text-transform:uppercase; font-size:.8rem; }
.section-header { color:#00ffb4; border-bottom:1px solid rgba(0,255,180,.18); padding-bottom:.5rem; margin:1.8rem 0 1rem; letter-spacing:.15em; }
.status-bar { color:#00ffb4; background:rgba(0,255,180,.06); border-left:3px solid #00ffb4; padding:.7rem 1rem; border-radius:0 8px 8px 0; margin:.8rem 0; }
.status-warn { color:#ff6666; background:rgba(255,68,68,.08); border-left-color:#ff4444; }
.metric-card { background:rgba(255,255,255,.025); border:1px solid rgba(0,255,180,.12); border-radius:10px; padding:1rem; text-align:center; }
.metric-num { font-size:2rem; font-weight:900; color:#00ffb4; }
.metric-label { color:#7b8da3; font-size:.75rem; letter-spacing:.08em; }
div[data-testid="stSidebar"] { background:#02060f; }
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.markdown("### ⚙ Параметры")
    confidence_threshold = st.slider("Порог уверенности (%)", 50, 99, 65)
    show_proba = st.checkbox("Показать вероятности классов", value=True)
    show_full_table = st.checkbox("Показать полную таблицу", value=True)
    st.markdown("---")
    st.caption("Поддерживает CSV и ZIP с CSV. Если в датасете есть колонка label/Label/class/target, модель обучается на загруженном датасете и показывает реальный вывод.")

# ---------------------- FUNCTIONS ----------------------
def load_data(uploaded_file):
    """Reads CSV or the first CSV inside ZIP."""
    if uploaded_file.name.lower().endswith(".zip"):
        content = uploaded_file.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                return None, "ZIP не содержит CSV-файлов"
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f), csv_files[0]
    return pd.read_csv(uploaded_file), uploaded_file.name

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

def find_label_column(df):
    possible = ["label", "Label", "target", "Target", "class", "Class", "attack", "Attack"]
    for col in possible:
        if col in df.columns:
            return col
    return None

def prepare_uploaded_dataset(df):
    """
    Universal preprocessing:
    - finds label column if it exists;
    - converts bool columns to 0/1;
    - one-hot encodes text columns;
    - fills missing numeric values.
    """
    df = clean_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    label_col = find_label_column(df)

    y = None
    if label_col:
        y = df[label_col].copy()
        X = df.drop(columns=[label_col])
    else:
        X = df.copy()

    # Convert boolean to integers
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # One-hot encode text/categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Make everything numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    return df, X, y, label_col

def train_on_uploaded_data(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    # If dataset is too small or has one class, cannot train normally
    if len(np.unique(y_enc)) < 2:
        return None, None, None, "В label найден только один класс. Нужны минимум 2 класса."

    stratify = y_enc if min(np.bincount(y_enc)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE only when each class has enough samples
    try:
        if min(np.bincount(y_train)) >= 6:
            X_train_scaled, y_train = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    except Exception:
        pass

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    test_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_pred)

    return model, scaler, le, {
        "accuracy": acc,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "features": list(X.columns)
    }

def normalize_label_name(x):
    s = str(x).strip()
    if s in ["0", "0.0", "Normal", "normal", "BENIGN", "Benign"]:
        return "Normal"
    if s in ["1", "1.0", "Anomaly", "anomaly", "Attack", "attack", "Malicious"]:
        return "Anomaly"
    return s

def make_pie(counts):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    fig.patch.set_alpha(0)
    return fig

# ---------------------- HERO ----------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡 IoT THREAT SHIELD</div>
    <div class="hero-sub">Система обнаружения аномалий в сетевом трафике IoT-устройств</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">▶ ЗАГРУЗКА ДАТАСЕТА</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Перетащите ZIP-архив или CSV-файл", type=["zip", "csv"], label_visibility="collapsed")

if uploaded is None:
    st.info("Загрузите ZIP или CSV. Для твоего файла archive (8).zip приложение само найдет embedded_system_network_security_dataset.csv.")
    st.stop()

df_raw, fname = load_data(uploaded)
if df_raw is None:
    st.error(fname)
    st.stop()

st.markdown(
    f'<div class="status-bar">📄 {fname} · {len(df_raw):,} записей · {len(df_raw.columns)} колонок</div>',
    unsafe_allow_html=True
)

df_clean, X, y, label_col = prepare_uploaded_dataset(df_raw)

if label_col is None:
    st.error(
        "В датасете нет колонки label/Label/target/class. "
        "Добавьте колонку с реальным классом атаки или нормального трафика."
    )
    st.write("Доступные колонки:", list(df_clean.columns))
    st.stop()

model, scaler, le, info = train_on_uploaded_data(X, y)

if isinstance(info, str):
    st.error(info)
    st.stop()

st.markdown(
    f'<div class="status-bar">✅ Модель обучена на загруженном датасете | features: {len(info["features"])} | train: {info["train_size"]} | test: {info["test_size"]} | accuracy: {info["accuracy"]*100:.2f}%</div>',
    unsafe_allow_html=True
)

X_scaled_all = scaler.transform(X)
proba = model.predict_proba(X_scaled_all)
pred = model.predict(X_scaled_all)
pred_labels_raw = le.inverse_transform(pred)

max_conf = proba.max(axis=1) * 100

result = df_clean.copy()
result["Реальный_класс"] = y.astype(str).map(normalize_label_name)
result["Предсказание"] = pd.Series(pred_labels_raw).map(normalize_label_name)
result["Уверенность_%"] = max_conf.round(2)
result["Статус"] = np.where(max_conf >= confidence_threshold, result["Предсказание"], "Неопределено")

for i, cls in enumerate(le.classes_):
    result[f"P({normalize_label_name(cls)})%"] = (proba[:, i] * 100).round(1)

# ---------------------- SUMMARY ----------------------
st.markdown('<div class="section-header">▶ РЕАЛЬНЫЙ ВЫВОД МОДЕЛИ</div>', unsafe_allow_html=True)

total = len(result)
normal_count = (result["Статус"] == "Normal").sum()
anomaly_count = (result["Статус"] == "Anomaly").sum()
undef_count = (result["Статус"] == "Неопределено").sum()
accuracy_all = (result["Реальный_класс"] == result["Предсказание"]).mean() * 100

c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, total, "ВСЕГО ЗАПИСЕЙ"),
    (c2, normal_count, "NORMAL"),
    (c3, anomaly_count, "ANOMALY"),
    (c4, f"{accuracy_all:.1f}%", "ACCURACY НА ФАЙЛЕ"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">▶ ГРАФИКИ</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.write("**Распределение предсказаний**")
    st.pyplot(make_pie(result["Статус"].value_counts()))
with col2:
    st.write("**Важность признаков**")
    imp = pd.Series(model.feature_importances_, index=info["features"]).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
    ax.barh(imp.index[::-1], imp.values[::-1])
    ax.set_xlabel("Importance")
    fig.patch.set_alpha(0)
    st.pyplot(fig)

st.markdown('<div class="section-header">▶ ОБНАРУЖЕННЫЕ АНОМАЛИИ</div>', unsafe_allow_html=True)
anomalies = result[result["Статус"].isin(["Anomaly"])].copy()

if len(anomalies) == 0:
    st.markdown('<div class="status-bar">✅ Аномалий не обнаружено</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="status-bar status-warn">⚠ Обнаружено {len(anomalies):,} аномальных записей</div>',
        unsafe_allow_html=True
    )
    display_cols = ["Статус", "Реальный_класс", "Предсказание", "Уверенность_%"]
    if show_proba:
        display_cols += [c for c in result.columns if c.startswith("P(")]
    display_cols += [c for c in df_clean.columns if c != label_col][:10]
    st.dataframe(anomalies[display_cols], use_container_width=True, height=420)

    st.download_button(
        "⬇ Скачать найденные аномалии CSV",
        anomalies.to_csv(index=False).encode("utf-8"),
        "anomalies.csv",
        "text/csv"
    )

if show_full_table:
    with st.expander("📋 Полная таблица с результатами"):
        display_cols = ["Статус", "Реальный_класс", "Предсказание", "Уверенность_%"]
        if show_proba:
            display_cols += [c for c in result.columns if c.startswith("P(")]
        display_cols += [c for c in df_clean.columns if c != label_col]
        st.dataframe(result[display_cols], use_container_width=True, height=450)

st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#365064; font-size:.7rem;">
IoT Threat Shield · Streamlit · Random Forest
</div>
""", unsafe_allow_html=True)
