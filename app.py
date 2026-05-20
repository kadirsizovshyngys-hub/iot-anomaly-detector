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
# IoT-SPECIFIC THRESHOLDS (based on IoT device characteristics)
# ─────────────────────────────────────────────────────────────────────────────
IOT_THRESHOLDS = {
    # Normal IoT traffic boundaries
    'normal': {
        'Packets':        (1, 100),       # IoT devices send few packets
        'Bytes':          (40, 2000),     # Small payloads (sensors, MQTT)
        'Flow_Duration':  (0.001, 5.0),   # Short flows typical for IoT
        'Packet_Rate':    (0, 500),       # Low packet rate
        'Byte_Rate':      (0, 10000),     # Low bandwidth
        'Packet_Size_Avg':(20, 200),      # Small avg packet size
    },
    # DDoS indicators
    'ddos': {
        'Packets':        5000,    # >> normal; flood of packets
        'Bytes':          100000,  # >> normal; massive volume
        'Packet_Rate':    1000,    # >> normal; high frequency
        'Byte_Rate':      50000,   # >> normal; bandwidth saturation
    },
    # PortScan indicators
    'portscan': {
        'Flow_Duration':  0.2,     # << normal; very short probing flows
        'Packets':        4,       # minimal packets (SYN only)
        'Packet_Size_Avg':60,      # tiny packets (SYN/ACK headers)
        'Byte_Rate':      500,     # low rate but many connections
    }
}

PROTOCOL_MAP = {0: 'HOPOPT', 1: 'ICMP', 6: 'TCP', 17: 'UDP', 58: 'ICMPv6'}

def get_protocol_name(val):
    try:
        return PROTOCOL_MAP.get(int(val), f'Proto-{int(val)}')
    except:
        return str(val)

# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION ENGINE — why is this row an anomaly?
# ─────────────────────────────────────────────────────────────────────────────
def explain_anomaly(row, label):
    reasons = []
    thr = IOT_THRESHOLDS

    if label == 'DDoS':
        if row.get('Packets', 0) > thr['ddos']['Packets']:
            reasons.append(f"Packets={int(row['Packets']):,} (норма IoT: <100)")
        if row.get('Bytes', 0) > thr['ddos']['Bytes']:
            reasons.append(f"Bytes={int(row['Bytes']):,} (норма IoT: <2000)")
        if row.get('Packet_Rate', 0) > thr['ddos']['Packet_Rate']:
            reasons.append(f"Packet_Rate={row['Packet_Rate']:.0f}/с (норма: <500)")
        if row.get('Byte_Rate', 0) > thr['ddos']['Byte_Rate']:
            reasons.append(f"Byte_Rate={row['Byte_Rate']:.0f} Б/с (норма: <10000)")
        if not reasons:
            reasons.append("Аномальная комбинация признаков трафика DDoS")

    elif label == 'PortScan':
        if row.get('Flow_Duration', 99) < thr['portscan']['Flow_Duration']:
            reasons.append(f"Flow_Duration={row['Flow_Duration']:.4f}с (норма: >0.001с, здесь слишком короткий)")
        if row.get('Packets', 99) <= thr['portscan']['Packets']:
            reasons.append(f"Packets={int(row['Packets'])} (зондирующий пакет SYN)")
        if row.get('Packet_Size_Avg', 999) < thr['portscan']['Packet_Size_Avg']:
            reasons.append(f"Avg размер пакета={row['Packet_Size_Avg']:.1f} Б (только заголовки TCP)")
        proto = get_protocol_name(row.get('Protocol', -1))
        if proto == 'TCP':
            reasons.append("Протокол TCP — типичен для сканирования портов")
        if not reasons:
            reasons.append("Паттерн короткого зондирующего соединения")

    return reasons

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING (IoT-calibrated synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    np.random.seed(42)
    n = 8000

    normal = pd.DataFrame({
        'Flow_Duration': np.random.exponential(0.5, n // 3).clip(0.001, 5.0),
        'Packets':       np.random.randint(1, 100, n // 3),
        'Bytes':         np.random.randint(40, 2000, n // 3),
        'Protocol':      np.random.choice([6, 17, 1], n // 3, p=[0.5, 0.4, 0.1]),
        'Label': 'Normal'
    })
    ddos = pd.DataFrame({
        'Flow_Duration': np.random.uniform(1.0, 60.0, n // 3),
        'Packets':       np.random.randint(5000, 80000, n // 3),
        'Bytes':         np.random.randint(100000, 5000000, n // 3),
        'Protocol':      np.random.choice([6, 17, 1], n // 3, p=[0.4, 0.4, 0.2]),
        'Label': 'DDoS'
    })
    portscan = pd.DataFrame({
        'Flow_Duration': np.random.uniform(0.0001, 0.15, n // 3),
        'Packets':       np.random.randint(1, 4, n // 3),
        'Bytes':         np.random.randint(40, 120, n // 3),
        'Protocol':      np.random.choice([6, 17], n // 3, p=[0.85, 0.15]),
        'Label': 'PortScan'
    })

    data = pd.concat([normal, ddos, portscan], ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    le = LabelEncoder()
    data['Label_enc'] = le.fit_transform(data['Label'])

    eps = 1e-9
    data['Packet_Rate']     = data['Packets'] / (data['Flow_Duration'] + eps)
    data['Byte_Rate']       = data['Bytes']   / (data['Flow_Duration'] + eps)
    data['Packet_Size_Avg'] = data['Bytes']   / (data['Packets'] + eps)

    feature_cols = ['Flow_Duration', 'Packets', 'Bytes', 'Protocol',
                    'Packet_Rate', 'Byte_Rate', 'Packet_Size_Avg']

    X = data[feature_cols]
    y = data['Label_enc']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # IoT-tuned Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y_res)
    return rf, scaler, le, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None, "ZIP не содержит CSV-файлов"
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f), csv_files[0]
    else:
        return pd.read_csv(uploaded_file), uploaded_file.name

def preprocess(df, scaler, feature_cols):
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    eps = 1e-9
    if 'Packet_Rate' not in df.columns:
        df['Packet_Rate']     = df['Packets'] / (df['Flow_Duration'] + eps)
    if 'Byte_Rate' not in df.columns:
        df['Byte_Rate']       = df['Bytes']   / (df['Flow_Duration'] + eps)
    if 'Packet_Size_Avg' not in df.columns:
        df['Packet_Size_Avg'] = df['Bytes']   / (df['Packets'] + eps)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None, None, f"Отсутствуют колонки: {missing}"
    X = df[feature_cols].copy()
    return df, scaler.transform(X), None

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'Normal': '#00ffb4', 'DDoS': '#ff4444', 'PortScan': '#ff8c00', 'Неопределено': '#4488ff'}

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor='none')
    ax.set_facecolor('#040810')
    for spine in ax.spines.values():
        spine.set_color('#1a2535')
    ax.tick_params(colors='#5a7a8a', labelsize=9)
    return fig, ax

def pie_chart(counts):
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='none')
    colors = [COLORS.get(l, '#888') for l in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'linewidth': 2, 'edgecolor': '#040810'},
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color('#040810')
        at.set_fontsize(9)
        at.set_fontweight('bold')
        at.set_fontfamily('monospace')
    legend_patches = [mpatches.Patch(color=COLORS.get(l, '#888'), label=l) for l in counts.index]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=False, labelcolor='#8b9ab0', fontsize=8)
    fig.patch.set_alpha(0)
    return fig

def bar_chart(importances, feature_cols):
    fig, ax = dark_fig(7, 3.2)
    imp = pd.Series(importances, index=feature_cols).sort_values()
    colors = ['#00ffb4' if v > imp.median() else '#1a3a4a' for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors, height=0.6, edgecolor='none')
    ax.set_xlabel('Важность', color='#5a7a8a', fontsize=8)
    ax.xaxis.label.set_fontfamily('monospace')
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#5a7a8a', fontsize=7,
                fontfamily='monospace')
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontfamily='monospace', fontsize=9)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

def hist_chart(df):
    fig, ax = dark_fig(7, 3.2)
    for label, color in COLORS.items():
        subset = df[df['Предсказание'] == label]['Уверенность (%)']
        if len(subset) > 0:
            ax.hist(subset, bins=25, alpha=0.75, label=label, color=color, edgecolor='none')
    ax.set_xlabel('Уверенность (%)', color='#5a7a8a', fontsize=8)
    ax.set_ylabel('Кол-во записей', color='#5a7a8a', fontsize=8)
    ax.legend(frameon=False, labelcolor='#8b9ab0', fontsize=8)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

def timeline_chart(df):
    """Show anomaly distribution across dataset rows."""
    fig, ax = dark_fig(10, 2.5)
    x = np.arange(len(df))
    colors = [COLORS.get(l, '#1a2535') for l in df['Предсказание']]
    ax.bar(x, 1, color=colors, width=1, edgecolor='none', alpha=0.9)
    ax.set_xlim(0, len(df))
    ax.set_yticks([])
    ax.set_xlabel('Индекс записи', color='#5a7a8a', fontsize=8)
    ax.set_title('Распределение аномалий по потоку трафика', color='#5a7a8a',
                 fontsize=9, fontfamily='monospace')
    legend_patches = [mpatches.Patch(color=COLORS[l], label=l)
                      for l in ['Normal', 'DDoS', 'PortScan'] if l in df['Предсказание'].values]
    ax.legend(handles=legend_patches, frameon=False, labelcolor='#8b9ab0',
              fontsize=8, loc='upper right')
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.8rem;
                color:#00ffb4; letter-spacing:0.2em; margin-bottom:1rem;">
        ⚙ ПАРАМЕТРЫ
    </div>
    """, unsafe_allow_html=True)

    confidence_threshold = st.slider("Порог уверенности (%)", 50, 99, 65)
    show_reasons = st.checkbox("Показывать причины аномалий", value=True)
    show_proba = st.checkbox("Вероятности классов", value=True)
    show_timeline = st.checkbox("Timeline трафика", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.72rem;
                color:#00ffb4; letter-spacing:0.15em; margin-bottom:0.8rem;">
        📡 IoT ПОРОГИ ДЕТЕКЦИИ
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rule-card rule-normal">
        <div class="rule-title">✅ Нормальный IoT трафик</div>
        <div class="rule-param">
            Пакеты: <span class="rule-val">1 – 100</span><br>
            Байты: <span class="rule-val">40 – 2,000</span><br>
            Длит. потока: <span class="rule-val">0.001 – 5.0 с</span><br>
            Протоколы: <span class="rule-val">TCP / UDP / ICMP</span><br>
            Avg размер пакета: <span class="rule-val">20 – 200 Б</span>
        </div>
    </div>
    <div class="rule-card rule-ddos">
        <div class="rule-title">🔴 DDoS атака</div>
        <div class="rule-param">
            Пакеты: <span class="rule-val">> 5,000</span> (флуд)<br>
            Байты: <span class="rule-val">> 100,000</span> (объём)<br>
            Packet Rate: <span class="rule-val">> 1,000 /с</span><br>
            Byte Rate: <span class="rule-val">> 50,000 Б/с</span><br>
            Цель: <span class="rule-val">перегрузка канала</span>
        </div>
    </div>
    <div class="rule-card rule-portscan">
        <div class="rule-title">🟠 Сканирование портов</div>
        <div class="rule-param">
            Длит. потока: <span class="rule-val">< 0.2 с</span> (зонд)<br>
            Пакеты: <span class="rule-val">1 – 3</span> (SYN)<br>
            Avg размер: <span class="rule-val">< 60 Б</span> (заголовок)<br>
            Протокол: <span class="rule-val">TCP (SYN scan)</span><br>
            Цель: <span class="rule-val">разведка сети</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem; color:#3d5a6e; letter-spacing:0.05em;">
        Модель: Random Forest (300 деревьев)<br>
        Балансировка: SMOTE<br>
        Обучающих образцов: 8,000<br>
        Классы: Normal · DDoS · PortScan
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡 IoT THREAT SHIELD</div>
    <div class="hero-line"></div>
    <div class="hero-sub">Система обнаружения аномалий в сетевом трафике IoT-устройств</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Инициализация модели..."):
    rf_model, scaler, le, feature_cols = train_model()

st.markdown('<div class="status-bar">✅ Модель инициализирована | Random Forest 300 деревьев | IoT-калиброванные пороги</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ЗАГРУЗКА ДАТАСЕТА</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Перетащите ZIP-архив или CSV-файл",
    type=["zip", "csv"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem; color:#1a3a4a;">
        <div style="font-size:3.5rem; margin-bottom:1rem; filter:drop-shadow(0 0 20px #00ffb4);">📡</div>
        <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00ffb4; letter-spacing:0.1em;">
            ОЖИДАНИЕ ДАННЫХ
        </div>
        <div style="margin-top:0.8rem; font-size:0.75rem; color:#2a4a5a;">
            Загрузите ZIP-архив с CSV-файлом трафика IoT<br>
            Ожидаемые колонки: Flow_Duration · Packets · Bytes · Protocol
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────────────────────────────────────
df_raw, fname = load_data(uploaded)
if df_raw is None:
    st.error(fname)
    st.stop()

st.markdown(f'<div class="status-bar">📄 {fname} · {len(df_raw):,} записей · {len(df_raw.columns)} колонок</div>',
            unsafe_allow_html=True)

df, X_scaled, err = preprocess(df_raw.copy(), scaler, feature_cols)
if err:
    st.error(err)
    st.stop()

# Predict
proba      = rf_model.predict_proba(X_scaled)
preds      = rf_model.predict(X_scaled)
pred_labels = le.inverse_transform(preds)
max_conf   = proba.max(axis=1) * 100

df['Предсказание']  = pred_labels
df['Уверенность (%)'] = max_conf.round(2)
df['Статус'] = np.where(max_conf >= confidence_threshold, pred_labels, 'Неопределено')

# Add per-class probabilities
for i, cls in enumerate(le.classes_):
    df[f'P({cls})%'] = (proba[:, i] * 100).round(1)

# Explain anomalies
if show_reasons:
    def get_reasons(row):
        if row['Статус'] in ('DDoS', 'PortScan'):
            return ' | '.join(explain_anomaly(row, row['Статус']))
        return ''
    df['Причина'] = df.apply(get_reasons, axis=1)

# Protocol name
df['Протокол'] = df['Protocol'].apply(get_protocol_name)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ СВОДКА АНАЛИЗА</div>', unsafe_allow_html=True)

total   = len(df)
n_norm  = (df['Статус'] == 'Normal').sum()
n_ddos  = (df['Статус'] == 'DDoS').sum()
n_port  = (df['Статус'] == 'PortScan').sum()
n_undef = (df['Статус'] == 'Неопределено').sum()
threat_pct = ((n_ddos + n_port) / total * 100) if total else 0

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, 'card-total',    'num-total',    total,        'ВСЕГО ЗАПИСЕЙ'),
    (c2, 'card-normal',   'num-normal',   n_norm,       'НОРМАЛЬНЫЙ'),
    (c3, 'card-ddos',     'num-ddos',     n_ddos,       'DDoS АТАК'),
    (c4, 'card-portscan', 'num-portscan', n_port,       'PORTSCAN'),
    (c5, 'card-ddos',     'num-ddos',     f'{threat_pct:.1f}%', 'УГРОЗ'),
]
for col, card_cls, num_cls, val, label in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card {card_cls}">
            <div class="metric-num {num_cls}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ВИЗУАЛИЗАЦИЯ</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1.2, 1.8, 1.8])

with col_a:
    st.markdown("**Классификация трафика**")
    counts = df['Статус'].value_counts()
    st.pyplot(pie_chart(counts))

with col_b:
    st.markdown("**Важность признаков (RF)**")
    st.pyplot(bar_chart(rf_model.feature_importances_, feature_cols))

with col_c:
    st.markdown("**Распределение уверенности**")
    st.pyplot(hist_chart(df))

if show_timeline:
    st.markdown("**Timeline трафика**")
    st.pyplot(timeline_chart(df))

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ОБНАРУЖЕННЫЕ УГРОЗЫ</div>', unsafe_allow_html=True)

anomalies = df[df['Статус'].isin(['DDoS', 'PortScan'])].copy()

if len(anomalies) == 0:
    st.markdown('<div class="status-bar">✅ Угроз не обнаружено — трафик в норме</div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="status-bar status-warn">⚠ ОБНАРУЖЕНО {len(anomalies):,} АНОМАЛЬНЫХ ПОТОКОВ '
        f'({threat_pct:.1f}% трафика) — требуется реакция</div>',
        unsafe_allow_html=True
    )

    # Build display columns
    display_cols = ['Статус', 'Уверенность (%)', 'Протокол',
                    'Flow_Duration', 'Packets', 'Bytes',
                    'Packet_Rate', 'Byte_Rate', 'Packet_Size_Avg']
    if show_proba:
        display_cols += [f'P({c})%' for c in le.classes_]
    if show_reasons:
        display_cols.append('Причина')

    existing = [c for c in display_cols if c in anomalies.columns]

    def highlight(row):
        base = 'background-color: '
        if row['Статус'] == 'DDoS':
            return [base + '#2a0808'] * len(row)
        elif row['Статус'] == 'PortScan':
            return [base + '#2a1400'] * len(row)
        return [''] * len(row)

    styled = (anomalies[existing]
              .reset_index(drop=True)
              .style
              .apply(highlight, axis=1)
              .format({
                  'Уверенность (%)': '{:.1f}%',
                  'Flow_Duration':   '{:.4f}',
                  'Packet_Rate':     '{:.1f}',
                  'Byte_Rate':       '{:.1f}',
                  'Packet_Size_Avg': '{:.1f}',
              }))

    st.dataframe(styled, use_container_width=True, height=420)

    # Download
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv_out = anomalies[existing].to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Скачать аномалии (CSV)", csv_out,
                           "anomalies.csv", "text/csv")
    with col_d2:
        # DDoS only
        ddos_only = anomalies[anomalies['Статус'] == 'DDoS'][existing]
        if len(ddos_only):
            st.download_button("⬇ Только DDoS (CSV)",
                               ddos_only.to_csv(index=False).encode('utf-8'),
                               "ddos_only.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# FULL TABLE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Полная таблица всех записей"):
    full_cols = ['Статус', 'Уверенность (%)', 'Протокол',
                 'Flow_Duration', 'Packets', 'Bytes', 'Packet_Rate']
    if show_reasons:
        full_cols.append('Причина')
    existing_full = [c for c in full_cols if c in df.columns]
    st.dataframe(df[existing_full], use_container_width=True, height=350)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#1a3a4a;
            font-size:0.65rem; letter-spacing:0.1em; border-top:1px solid #0a1a2a; margin-top:2rem;">
    IoT Threat Shield · Дипломная работа · Обнаружение аномалий в трафике IoT-устройств
</div>
""", unsafe_allow_html=True)
