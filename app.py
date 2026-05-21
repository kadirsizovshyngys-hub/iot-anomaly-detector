import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Threat Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background: #040810;
    color: #c9d1d9;
}

.stApp { background: #040810; }

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,255,180,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,180,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 3rem;
    letter-spacing: 0.08em;
    color: #00ffb4;
    text-shadow: 0 0 30px rgba(0,255,180,0.5), 0 0 60px rgba(0,255,180,0.2);
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #3d5a6e;
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.hero-line {
    width: 200px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffb4, transparent);
    margin: 1rem auto;
}

/* Metric cards */
.metric-grid { display: flex; gap: 1rem; margin: 1.5rem 0; }
.metric-card {
    flex: 1;
    background: rgba(0,255,180,0.03);
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.card-normal::before  { background: #00ffb4; }
.card-ddos::before    { background: #ff4444; }
.card-portscan::before{ background: #ff8c00; }
.card-total::before   { background: #4488ff; }

.metric-num {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d5a6e;
    margin-top: 0.3rem;
}
.num-normal   { color: #00ffb4; }
.num-ddos     { color: #ff4444; text-shadow: 0 0 20px rgba(255,68,68,0.4); }
.num-portscan { color: #ff8c00; text-shadow: 0 0 20px rgba(255,140,0,0.4); }
.num-total    { color: #4488ff; }

/* Threat rule cards */
.rule-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.rule-title {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.rule-normal   { color: #00ffb4; border-left: 3px solid #00ffb4; }
.rule-ddos     { color: #ff4444; border-left: 3px solid #ff4444; }
.rule-portscan { color: #ff8c00; border-left: 3px solid #ff8c00; }
.rule-param {
    font-size: 0.68rem;
    color: #8b9ab0;
    line-height: 1.7;
}
.rule-val { color: #e2e8f0; font-weight: 700; }

/* Status bar */
.status-bar {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: #00ffb4;
    background: rgba(0,255,180,0.06);
    border-left: 3px solid #00ffb4;
    padding: 0.5rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.8rem 0;
}
.status-warn {
    color: #ff4444;
    background: rgba(255,68,68,0.06);
    border-left-color: #ff4444;
}

/* Reason tags */
.reason-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    margin: 1px 2px;
    font-weight: 700;
}
.tag-ddos     { background: rgba(255,68,68,0.15);  color: #ff6666; border: 1px solid rgba(255,68,68,0.3); }
.tag-portscan { background: rgba(255,140,0,0.15);  color: #ffaa33; border: 1px solid rgba(255,140,0,0.3); }
.tag-normal   { background: rgba(0,255,180,0.1);   color: #00ffb4; border: 1px solid rgba(0,255,180,0.2); }

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00ffb4;
    border-bottom: 1px solid rgba(0,255,180,0.15);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(0,255,180,0.03);
    border: 2px dashed rgba(0,255,180,0.2);
    border-radius: 10px;
    padding: 1rem;
    transition: border-color 0.3s;
}
div[data-testid="stSidebar"] {
    background: #02060f;
    border-right: 1px solid rgba(0,255,180,0.08);
}
.stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE DATASET ENGINE
# ─────────────────────────────────────────────────────────────────────────────
IOT_THRESHOLDS = {
    'normal': {'Packets': (1, 100), 'Bytes': (40, 2000), 'Flow_Duration': (0.001, 5.0), 'Packet_Rate': (0, 500), 'Byte_Rate': (0, 10000), 'Packet_Size_Avg': (20, 200)},
    'ddos': {'Packets': 5000, 'Bytes': 100000, 'Packet_Rate': 1000, 'Byte_Rate': 50000},
    'portscan': {'Flow_Duration': 0.2, 'Packets': 4, 'Packet_Size_Avg': 60, 'Byte_Rate': 500}
}
PROTOCOL_MAP = {0: 'HOPOPT', 1: 'ICMP', 6: 'TCP', 17: 'UDP', 58: 'ICMPv6'}
COLORS = {'Normal': '#00ffb4', 'DDoS': '#ff4444', 'PortScan': '#ff8c00', 'Anomaly': '#ff4444', 'Attack': '#ff4444', 'Неопределено': '#4488ff'}

def clean_name(s):
    return str(s).strip().replace(' ', '_').replace('-', '_')

def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                return None, 'ZIP не содержит CSV-файлов'
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f), csv_files[0]
    return pd.read_csv(uploaded_file), uploaded_file.name

def find_label_column(df):
    for c in df.columns:
        if c.lower() in ['label', 'target', 'class', 'attack', 'category', 'type']:
            return c
    return None

def normalize_protocol(df):
    # direct numeric/string protocol column
    for c in df.columns:
        lc = c.lower()
        if lc in ['protocol', 'proto'] or 'protocol' == lc:
            s = df[c]
            if s.dtype == object:
                return s.astype(str).str.upper().map({'TCP':6,'UDP':17,'ICMP':1,'ICMPV6':58}).fillna(0)
            return pd.to_numeric(s, errors='coerce').fillna(0)
    # one-hot protocol columns: protocol_type_TCP / protocol_TCP etc.
    proto_cols = [c for c in df.columns if 'protocol' in c.lower()]
    if proto_cols:
        out = pd.Series(0, index=df.index, dtype=float)
        for c in proto_cols:
            name = c.upper()
            val = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
            if 'TCP' in name: out = np.where(val > 0, 6, out)
            elif 'UDP' in name: out = np.where(val > 0, 17, out)
            elif 'ICMP' in name: out = np.where(val > 0, 1, out)
        return pd.Series(out, index=df.index)
    return pd.Series(0, index=df.index)

def first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(n.lower() in lc for n in names):
            return c
    return None

def add_universal_iot_features(df):
    """Creates the old app columns from many possible dataset schemas."""
    df = df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    # bool -> int
    for c in df.select_dtypes(include=['bool']).columns:
        df[c] = df[c].astype(int)

    pkt_col = first_col(df, ['Packets','packet_count','packet_count_5s','packets_total','total_packets','fwd_packets'])
    byte_col = first_col(df, ['Bytes','bytes','total_bytes','flow_bytes','packet_size_total'])
    dur_col = first_col(df, ['Flow_Duration','duration','flow_duration','inter_arrival_time','iat','time_delta'])
    size_col = first_col(df, ['Packet_Size_Avg','mean_packet_size','avg_packet_size','packet_size','packet_len','length'])

    if 'Packets' not in df.columns:
        df['Packets'] = pd.to_numeric(df[pkt_col], errors='coerce') if pkt_col else 1
    if 'Flow_Duration' not in df.columns:
        df['Flow_Duration'] = pd.to_numeric(df[dur_col], errors='coerce') if dur_col else 1.0
    if 'Packet_Size_Avg' not in df.columns:
        df['Packet_Size_Avg'] = pd.to_numeric(df[size_col], errors='coerce') if size_col else np.nan
    if 'Bytes' not in df.columns:
        if byte_col:
            df['Bytes'] = pd.to_numeric(df[byte_col], errors='coerce')
        else:
            df['Bytes'] = pd.to_numeric(df['Packets'], errors='coerce').fillna(1) * pd.to_numeric(df['Packet_Size_Avg'], errors='coerce').fillna(100)
    if 'Protocol' not in df.columns:
        df['Protocol'] = normalize_protocol(df)

    eps = 1e-9
    for c in ['Packets','Bytes','Flow_Duration','Packet_Size_Avg','Protocol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Packets'] = df['Packets'].fillna(df['Packets'].median()).fillna(1)
    df['Bytes'] = df['Bytes'].fillna(df['Bytes'].median()).fillna(100)
    df['Flow_Duration'] = df['Flow_Duration'].fillna(df['Flow_Duration'].median()).fillna(1)
    df['Packet_Size_Avg'] = df['Packet_Size_Avg'].fillna(df['Bytes']/(df['Packets']+eps)).fillna(100)
    df['Protocol'] = df['Protocol'].fillna(0)
    df['Packet_Rate'] = df['Packets'] / (df['Flow_Duration'].abs() + eps)
    df['Byte_Rate'] = df['Bytes'] / (df['Flow_Duration'].abs() + eps)
    return df

def make_ml_matrix(df, label_col=None):
    X = df.drop(columns=[label_col], errors='ignore').copy()
    for c in X.select_dtypes(include=['bool']).columns:
        X[c] = X[c].astype(int)
    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X

def train_model_from_uploaded(df, label_col):
    y_raw = df[label_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    if len(np.unique(y)) < 2:
        return None, None, None, None, 'В label только один класс — модель не может обучиться.'
    X = make_ml_matrix(df, label_col)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        if min(np.bincount(y)) >= 6:
            X_scaled, y = SMOTE(random_state=42).fit_resample(X_scaled, y)
    except Exception:
        pass
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    return rf, scaler, le, list(X.columns), None

@st.cache_resource(show_spinner=False)
def train_fallback_model():
    np.random.seed(42); n = 6000
    normal = pd.DataFrame({'Flow_Duration':np.random.exponential(.5,n//3).clip(.001,5), 'Packets':np.random.randint(1,100,n//3), 'Bytes':np.random.randint(40,2000,n//3), 'Protocol':np.random.choice([6,17,1],n//3), 'Label':'Normal'})
    ddos = pd.DataFrame({'Flow_Duration':np.random.uniform(1,60,n//3), 'Packets':np.random.randint(5000,80000,n//3), 'Bytes':np.random.randint(100000,5000000,n//3), 'Protocol':np.random.choice([6,17,1],n//3), 'Label':'DDoS'})
    ps = pd.DataFrame({'Flow_Duration':np.random.uniform(.0001,.15,n//3), 'Packets':np.random.randint(1,4,n//3), 'Bytes':np.random.randint(40,120,n//3), 'Protocol':np.random.choice([6,17],n//3,p=[.85,.15]), 'Label':'PortScan'})
    d = add_universal_iot_features(pd.concat([normal,ddos,ps], ignore_index=True))
    le=LabelEncoder(); y=le.fit_transform(d['Label']); features=['Flow_Duration','Packets','Bytes','Protocol','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    sc=StandardScaler(); X=sc.fit_transform(d[features])
    rf=RandomForestClassifier(n_estimators=300,max_depth=20,class_weight='balanced',random_state=42,n_jobs=-1).fit(X,y)
    return rf, sc, le, features

def get_protocol_name(val):
    try: return PROTOCOL_MAP.get(int(val), f'Proto-{int(val)}')
    except Exception: return str(val)

def normalize_label(label):
    s = str(label).strip()
    low = s.lower()
    if low in ['0','0.0','normal','benign','legitimate']: return 'Normal'
    if 'ddos' in low or 'dos' in low: return 'DDoS'
    if 'port' in low or 'scan' in low: return 'PortScan'
    if low in ['1','1.0','attack','anomaly','malicious','abnormal']: return 'Anomaly'
    return s

def explain_anomaly(row, label):
    reasons=[]; thr=IOT_THRESHOLDS
    # universal reasons based on old IoT thresholds, independent of original columns
    if row.get('Packets',0) > thr['ddos']['Packets']: reasons.append(f"Packets={row['Packets']:.0f} — слишком много пакетов для IoT")
    if row.get('Bytes',0) > thr['ddos']['Bytes']: reasons.append(f"Bytes={row['Bytes']:.0f} — слишком большой объём трафика")
    if row.get('Packet_Rate',0) > thr['ddos']['Packet_Rate']: reasons.append(f"Packet_Rate={row['Packet_Rate']:.1f}/с — высокая частота пакетов")
    if row.get('Byte_Rate',0) > thr['ddos']['Byte_Rate']: reasons.append(f"Byte_Rate={row['Byte_Rate']:.1f} Б/с — высокий поток байтов")
    if row.get('Flow_Duration',99) < thr['portscan']['Flow_Duration'] and row.get('Packets',99) <= thr['portscan']['Packets']:
        reasons.append(f"Очень короткий поток + мало пакетов — похоже на probing/PortScan")
    if row.get('Packet_Size_Avg',999) < thr['portscan']['Packet_Size_Avg']:
        reasons.append(f"Packet_Size_Avg={row['Packet_Size_Avg']:.2f} — маленький пакет, часто только заголовки")
    proto = get_protocol_name(row.get('Protocol',-1))
    if 'PortScan' in str(label) and proto == 'TCP': reasons.append('TCP-протокол типичен для SYN/port scanning')
    # dataset-specific flags
    for c in row.index:
        lc=c.lower()
        try: v=float(row[c])
        except Exception: continue
        if v>0 and any(k in lc for k in ['syn','fin','entropy','frequency','inter_arrival']):
            reasons.append(f"{c}={row[c]} — важный сетевой признак, связанный с аномальным паттерном")
    if not reasons: reasons.append('Аномалия определена по комбинации признаков модели')
    return ' | '.join(reasons[:4])

def dark_fig(w,h):
    fig, ax = plt.subplots(figsize=(w,h), facecolor='none'); ax.set_facecolor('#040810')
    for spine in ax.spines.values(): spine.set_color('#1a2535')
    ax.tick_params(colors='#5a7a8a', labelsize=9); return fig, ax

def pie_chart(counts):
    fig, ax = plt.subplots(figsize=(4.5,4.5), facecolor='none')
    colors=[COLORS.get(l,'#888') for l in counts.index]
    ax.pie(counts.values, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'linewidth':2,'edgecolor':'#040810'})
    ax.legend(handles=[mpatches.Patch(color=COLORS.get(l,'#888'), label=l) for l in counts.index], loc='lower center', bbox_to_anchor=(0.5,-.08), ncol=3, frameon=False, labelcolor='#8b9ab0', fontsize=8)
    fig.patch.set_alpha(0); return fig

def bar_chart(importances, feature_cols):
    fig, ax=dark_fig(7,3.2); imp=pd.Series(importances,index=feature_cols).sort_values().tail(15)
    ax.barh(imp.index, imp.values, color='#00ffb4'); ax.set_xlabel('Важность', color='#5a7a8a', fontsize=8); fig.patch.set_alpha(0); plt.tight_layout(); return fig

def hist_chart(df):
    fig, ax=dark_fig(7,3.2)
    for label in df['Статус'].unique(): ax.hist(df[df['Статус']==label]['Уверенность (%)'], bins=20, alpha=.7, label=label, color=COLORS.get(label,'#4488ff'))
    ax.set_xlabel('Уверенность (%)', color='#5a7a8a'); ax.legend(frameon=False, labelcolor='#8b9ab0', fontsize=8); fig.patch.set_alpha(0); plt.tight_layout(); return fig

def timeline_chart(df):
    fig, ax=dark_fig(10,2.5); x=np.arange(len(df)); colors=[COLORS.get(l,'#4488ff') for l in df['Статус']]
    ax.bar(x,1,color=colors,width=1,edgecolor='none'); ax.set_yticks([]); ax.set_xlabel('Индекс записи', color='#5a7a8a'); fig.patch.set_alpha(0); plt.tight_layout(); return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#00ffb4; letter-spacing:0.2em; margin-bottom:1rem;">⚙ ПАРАМЕТРЫ</div>
    """, unsafe_allow_html=True)
    confidence_threshold = st.slider('Порог уверенности (%)', 50, 99, 65)
    show_reasons = st.checkbox('Показывать причины аномалий', value=True)
    show_proba = st.checkbox('Вероятности классов', value=True)
    show_timeline = st.checkbox('Timeline трафика', value=True)
    st.markdown('---')
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.72rem; color:#00ffb4; letter-spacing:0.15em; margin-bottom:0.8rem;">📡 IoT ПОРОГИ ДЕТЕКЦИИ</div>
    <div class="rule-card rule-normal"><div class="rule-title">✅ Нормальный IoT трафик</div><div class="rule-param">Пакеты: <span class="rule-val">1–100</span><br>Байты: <span class="rule-val">40–2,000</span><br>Длит. потока: <span class="rule-val">0.001–5.0 c</span></div></div>
    <div class="rule-card rule-ddos"><div class="rule-title">🔴 DDoS атака</div><div class="rule-param">Пакеты: <span class="rule-val">>5,000</span><br>Байты: <span class="rule-val">>100,000</span><br>Packet Rate: <span class="rule-val">>1,000/с</span></div></div>
    <div class="rule-card rule-portscan"><div class="rule-title">🟠 PortScan</div><div class="rule-param">Короткий поток + малые пакеты + TCP/SYN признаки</div></div>
    """, unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero"><div class="hero-title">🛡 IoT THREAT SHIELD</div><div class="hero-line"></div><div class="hero-sub">Система обнаружения аномалий в сетевом трафике IoT-устройств</div></div>
""", unsafe_allow_html=True)
st.markdown('<div class="status-bar">✅ Адаптивный режим | ZIP/CSV | авто-колонки | причины аномалий сохранены</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">▶ ЗАГРУЗКА ДАТАСЕТА</div>', unsafe_allow_html=True)
uploaded = st.file_uploader('Перетащите ZIP-архив или CSV-файл', type=['zip','csv'], label_visibility='collapsed')
if uploaded is None:
    st.markdown('<div style="text-align:center; padding:4rem 1rem; color:#1a3a4a;"><div style="font-size:3.5rem;">📡</div><div style="font-family:Orbitron,monospace; font-size:1rem; color:#00ffb4; letter-spacing:0.1em;">ОЖИДАНИЕ ДАННЫХ</div><div style="margin-top:0.8rem; font-size:0.75rem; color:#2a4a5a;">Загрузите ZIP или CSV. Колонки будут адаптированы автоматически.</div></div>', unsafe_allow_html=True)
    st.stop()

df_raw, fname = load_data(uploaded)
if df_raw is None: st.error(fname); st.stop()
st.markdown(f'<div class="status-bar">📄 {fname} · {len(df_raw):,} записей · {len(df_raw.columns)} колонок</div>', unsafe_allow_html=True)

df = add_universal_iot_features(df_raw)
label_col = find_label_column(df)

if label_col:
    rf_model, scaler, le, feature_cols, err = train_model_from_uploaded(df, label_col)
    if err: st.error(err); st.stop()
    X = make_ml_matrix(df, label_col)
    # align columns used during training
    X = X.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(X)
    model_mode = f'модель обучена на загруженном датасете по колонке `{label_col}`'
else:
    rf_model, scaler, le, feature_cols = train_fallback_model()
    X_scaled = scaler.transform(df[feature_cols])
    model_mode = 'label не найден — используется встроенная IoT-модель'

st.markdown(f'<div class="status-bar">✅ {model_mode}</div>', unsafe_allow_html=True)

proba = rf_model.predict_proba(X_scaled)
preds = rf_model.predict(X_scaled)
pred_labels = [normalize_label(x) for x in le.inverse_transform(preds)]
max_conf = proba.max(axis=1) * 100

df['Предсказание'] = pred_labels
df['Уверенность (%)'] = max_conf.round(2)
df['Статус'] = np.where(max_conf >= confidence_threshold, df['Предсказание'], 'Неопределено')
for i, cls in enumerate(le.classes_): df[f'P({normalize_label(cls)})%'] = (proba[:,i]*100).round(1)
df['Протокол'] = df['Protocol'].apply(get_protocol_name)
if show_reasons:
    df['Причина'] = df.apply(lambda r: explain_anomaly(r, r['Статус']) if r['Статус'] not in ['Normal','Неопределено'] else '', axis=1)

st.markdown('<div class="section-header">▶ СВОДКА АНАЛИЗА</div>', unsafe_allow_html=True)
total=len(df); n_norm=(df['Статус']=='Normal').sum(); n_undef=(df['Статус']=='Неопределено').sum(); n_threat=total-n_norm-n_undef; threat_pct=(n_threat/total*100) if total else 0
classes = df['Статус'].value_counts()
cols=st.columns(5)
vals=[(total,'ВСЕГО ЗАПИСЕЙ','num-total'),(n_norm,'НОРМАЛЬНЫЙ','num-normal'),(n_threat,'АНОМАЛИИ','num-ddos'),(n_undef,'НЕОПРЕДЕЛЕНО','num-total'),(f'{threat_pct:.1f}%','УГРОЗ','num-ddos')]
for col,(val,lab,cls) in zip(cols,vals):
    with col: st.markdown(f'<div class="metric-card"><div class="metric-num {cls}">{val}</div><div class="metric-label">{lab}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">▶ ВИЗУАЛИЗАЦИЯ</div>', unsafe_allow_html=True)
c1,c2,c3=st.columns([1.2,1.8,1.8])
with c1: st.markdown('**Классификация трафика**'); st.pyplot(pie_chart(classes))
with c2: st.markdown('**Важность признаков (RF)**'); st.pyplot(bar_chart(rf_model.feature_importances_, feature_cols))
with c3: st.markdown('**Распределение уверенности**'); st.pyplot(hist_chart(df))
if show_timeline: st.markdown('**Timeline трафика**'); st.pyplot(timeline_chart(df))

st.markdown('<div class="section-header">▶ ОБНАРУЖЕННЫЕ УГРОЗЫ</div>', unsafe_allow_html=True)
anomalies = df[~df['Статус'].isin(['Normal','Неопределено'])].copy()
if len(anomalies)==0:
    st.markdown('<div class="status-bar">✅ Угроз не обнаружено — трафик в норме или ниже порога уверенности</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-bar status-warn">⚠ ОБНАРУЖЕНО {len(anomalies):,} АНОМАЛЬНЫХ ПОТОКОВ ({threat_pct:.1f}% трафика)</div>', unsafe_allow_html=True)
    display_cols=['Статус','Уверенность (%)','Протокол','Flow_Duration','Packets','Bytes','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    if show_proba: display_cols += [c for c in df.columns if c.startswith('P(')]
    if show_reasons: display_cols.append('Причина')
    display_cols=[c for c in display_cols if c in anomalies.columns]
    st.dataframe(anomalies[display_cols].reset_index(drop=True), use_container_width=True, height=420)
    st.download_button('⬇ Скачать аномалии (CSV)', anomalies[display_cols].to_csv(index=False).encode('utf-8'), 'anomalies.csv', 'text/csv')

with st.expander('📋 Полная таблица всех записей'):
    full_cols=['Статус','Уверенность (%)','Протокол','Flow_Duration','Packets','Bytes','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    if show_reasons: full_cols.append('Причина')
    st.dataframe(df[[c for c in full_cols if c in df.columns]], use_container_width=True, height=350)

st.markdown('<div style="text-align:center; padding:2rem 0 1rem; color:#1a3a4a; font-size:0.65rem; letter-spacing:0.1em; border-top:1px solid #0a1a2a; margin-top:2rem;">IoT Threat Shield · Дипломная работа · Обнаружение аномалий в трафике IoT-устройств</div>', unsafe_allow_html=True)
