import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

st.set_page_config(page_title="IoT Anomaly Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background: #040810; color: #c9d1d9; }
.stApp { background: #040810; }
.stApp::before {
    content: ''; position: fixed; top:0; left:0; right:0; bottom:0;
    background-image: linear-gradient(rgba(0,255,180,0.03) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,255,180,0.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
.hero-title { font-family:'Orbitron',monospace; font-weight:900; font-size:2.8rem; color:#00ffb4; text-shadow:0 0 30px rgba(0,255,180,0.5); text-align:center; margin:0; letter-spacing:0.08em; }
.hero-sub { text-align:center; font-size:0.72rem; letter-spacing:0.25em; color:#3d5a6e; margin-top:0.4rem; text-transform:uppercase; }
.hero-line { width:200px; height:1px; background:linear-gradient(90deg,transparent,#00ffb4,transparent); margin:1rem auto; }
.metric-card { background:rgba(0,255,180,0.03); border:1px solid rgba(0,255,180,0.12); border-radius:8px; padding:1.2rem; text-align:center; position:relative; overflow:hidden; }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.card-normal::before  { background:#00ffb4; }
.card-anomaly::before { background:#ff4444; }
.card-heat::before    { background:#ff8c00; }
.card-battery::before { background:#facc15; }
.card-total::before   { background:#4488ff; }
.metric-num { font-family:'Orbitron',monospace; font-size:2rem; font-weight:700; line-height:1; }
.metric-label { font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase; color:#3d5a6e; margin-top:0.3rem; }
.num-normal  { color:#00ffb4; }
.num-anomaly { color:#ff4444; text-shadow:0 0 20px rgba(255,68,68,0.4); }
.num-heat    { color:#ff8c00; }
.num-battery { color:#facc15; }
.num-total   { color:#4488ff; }
.rule-card { background:rgba(255,255,255,0.02); border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.6rem; }
.rule-normal  { border-left:3px solid #00ffb4; }
.rule-anomaly { border-left:3px solid #ff4444; }
.rule-title { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.4rem; }
.rule-param { font-size:0.68rem; color:#8b9ab0; line-height:1.8; }
.rule-val { color:#e2e8f0; font-weight:700; }
.status-ok   { font-size:0.72rem; color:#00ffb4; background:rgba(0,255,180,0.06); border-left:3px solid #00ffb4; padding:0.5rem 1rem; border-radius:0 6px 6px 0; margin:0.8rem 0; }
.status-warn { font-size:0.72rem; color:#ff4444; background:rgba(255,68,68,0.06); border-left:3px solid #ff4444; padding:0.5rem 1rem; border-radius:0 6px 6px 0; margin:0.8rem 0; }
.section-hdr { font-family:'Orbitron',monospace; font-size:0.85rem; letter-spacing:0.2em; text-transform:uppercase; color:#00ffb4; border-bottom:1px solid rgba(0,255,180,0.15); padding-bottom:0.5rem; margin:2rem 0 1rem; }
div[data-testid="stFileUploader"] { background:rgba(0,255,180,0.03); border:2px dashed rgba(0,255,180,0.2); border-radius:10px; padding:1rem; }
div[data-testid="stSidebar"] { background:#02060f; border-right:1px solid rgba(0,255,180,0.08); }
</style>
""", unsafe_allow_html=True)

THRESHOLDS = {
    'temp_normal_max':  49.0,
    'temp_anomaly_min': 50.0,
    'temp_critical':    65.0,
    'humidity_min':     20.0,
    'humidity_max':     90.0,
    'battery_low':      20,
    'battery_critical': 10,
    'battery_dead':      5,
}

def explain_anomaly(row, pred):
    reasons = []
    if pred == 1:
        t = row.get('Temperature', 0)
        b = row.get('Battery_Level', 100)
        h = row.get('Humidity', 50)
        if t >= THRESHOLDS['temp_critical']:
            reasons.append(f"🔥 Критический перегрев: {t:.1f}°C (норма <{THRESHOLDS['temp_normal_max']}°C)")
        elif t >= THRESHOLDS['temp_anomaly_min']:
            reasons.append(f"♨️ Перегрев датчика: {t:.1f}°C (норма <{THRESHOLDS['temp_normal_max']}°C)")
        if b <= THRESHOLDS['battery_dead']:
            reasons.append(f"🪫 Батарея почти разряжена: {b}%")
        elif b <= THRESHOLDS['battery_critical']:
            reasons.append(f"🔋 Критический уровень батареи: {b}%")
        if h < THRESHOLDS['humidity_min']:
            reasons.append(f"💨 Аномально низкая влажность: {h:.1f}%")
        elif h > THRESHOLDS['humidity_max']:
            reasons.append(f"💧 Аномально высокая влажность: {h:.1f}%")
        if not reasons:
            reasons.append(f"⚠️ Аномальная комбинация (T={t:.1f}°C, H={h:.1f}%, B={b}%)")
    return ' | '.join(reasons)

def anomaly_type(row, pred):
    if pred == 0:
        return 'Normal'
    t = row.get('Temperature', 0)
    b = row.get('Battery_Level', 100)
    if t >= THRESHOLDS['temp_anomaly_min'] and b <= THRESHOLDS['battery_critical']:
        return 'Перегрев + Разряд'
    elif t >= THRESHOLDS['temp_anomaly_min']:
        return 'Перегрев'
    elif b <= THRESHOLDS['battery_critical']:
        return 'Критический разряд'
    return 'Аномалия датчика'

@st.cache_resource(show_spinner=False)
def train_model():
    np.random.seed(42)
    n = 6000

    normal = pd.DataFrame({
        'Temperature':   np.random.normal(30.0, 5.0, int(n*0.9)).clip(13, 49),
        'Humidity':      np.random.normal(50.0, 10.0, int(n*0.9)).clip(20, 89),
        'Battery_Level': np.random.randint(20, 100, int(n*0.9)),
        'Label': 0
    })
    anomaly = pd.DataFrame({
        'Temperature':   np.random.normal(60.0, 5.0, int(n*0.1)).clip(41, 73),
        'Humidity':      np.random.normal(48.5, 9.6, int(n*0.1)).clip(20, 80),
        'Battery_Level': np.random.randint(0, 20, int(n*0.1)),
        'Label': 1
    })

    data = pd.concat([normal, anomaly], ignore_index=True)
    data['Temp_Battery_Ratio'] = data['Temperature'] / (data['Battery_Level'] + 1)
    data['Heat_Risk']          = (data['Temperature'] > THRESHOLDS['temp_anomaly_min']).astype(int)
    data['Battery_Critical']   = (data['Battery_Level'] < THRESHOLDS['battery_critical']).astype(int)

    feature_cols = ['Temperature', 'Humidity', 'Battery_Level',
                    'Temp_Battery_Ratio', 'Heat_Risk', 'Battery_Critical']

    X = data[feature_cols]
    y = data['Label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Manual oversampling instead of SMOTE
    train_df = X_train.copy()
    train_df['Label'] = y_train.values
    majority = train_df[train_df['Label'] == 0]
    minority = train_df[train_df['Label'] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced = pd.concat([majority, minority_upsampled])
    X_bal = balanced[feature_cols]
    y_bal = balanced['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y_bal)
    return rf, scaler, feature_cols

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return None, "ZIP не содержит CSV-файлов"
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f), csv_files[0]
    return pd.read_csv(uploaded_file), uploaded_file.name

def preprocess(df, scaler, feature_cols):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    required = ['Temperature', 'Humidity', 'Battery_Level']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, None, f"Отсутствуют колонки: {missing}. Доступные: {list(df.columns)}"
    df['Temp_Battery_Ratio'] = df['Temperature'] / (df['Battery_Level'] + 1)
    df['Heat_Risk']          = (df['Temperature'] > THRESHOLDS['temp_anomaly_min']).astype(int)
    df['Battery_Critical']   = (df['Battery_Level'] < THRESHOLDS['battery_critical']).astype(int)
    X = df[feature_cols].copy()
    return df, scaler.transform(X), None

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor='none')
    ax.set_facecolor('#040810')
    for spine in ax.spines.values():
        spine.set_color('#1a2535')
    ax.tick_params(colors='#5a7a8a', labelsize=9)
    return fig, ax

COLORS = {'Normal':'#00ffb4','Перегрев':'#ff4444','Критический разряд':'#facc15',
          'Перегрев + Разряд':'#ff8c00','Аномалия датчика':'#c084fc'}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.8rem;color:#00ffb4;letter-spacing:0.2em;margin-bottom:1rem;">⚙ ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider("Порог уверенности (%)", 50, 99, 65)
    show_reasons  = st.checkbox("Причины аномалий", value=True)
    show_timeline = st.checkbox("График по устройствам", value=True)

    st.markdown("---")
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.72rem;color:#00ffb4;letter-spacing:0.15em;margin-bottom:0.8rem;">📡 ПОРОГИ IoT-СЕНСОРОВ</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rule-card rule-normal">
        <div class="rule-title" style="color:#00ffb4;">✅ Нормальная работа</div>
        <div class="rule-param">
            Температура: <span class="rule-val">13 – 49°C</span><br>
            Влажность: <span class="rule-val">20 – 89%</span><br>
            Заряд батареи: <span class="rule-val">20 – 100%</span>
        </div>
    </div>
    <div class="rule-card rule-anomaly">
        <div class="rule-title" style="color:#ff8c00;">🔥 Перегрев датчика</div>
        <div class="rule-param">
            Температура <span class="rule-val">> 50°C</span> — перегрев<br>
            Температура <span class="rule-val">> 65°C</span> — критично<br>
            Причина: сбой охлаждения, пожар
        </div>
    </div>
    <div class="rule-card rule-anomaly">
        <div class="rule-title" style="color:#facc15;">🪫 Критический разряд</div>
        <div class="rule-param">
            Батарея <span class="rule-val">< 10%</span> — аномалия<br>
            Батарея <span class="rule-val">< 5%</span> — отключение<br>
            Причина: утечка тока, старение
        </div>
    </div>
    <div class="rule-card rule-anomaly">
        <div class="rule-title" style="color:#c084fc;">💧 Аномалия влажности</div>
        <div class="rule-param">
            Влажность <span class="rule-val">< 20%</span> — слишком сухо<br>
            Влажность <span class="rule-val">> 90%</span> — конденсат<br>
            Причина: неисправность датчика
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:#3d5a6e;">Модель: Random Forest 300 деревьев<br>Балансировка: Oversampling<br>Признаки: Temperature, Humidity,<br>Battery_Level + инженерные</div>', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1rem;">
    <div class="hero-title">🛡 IoT ANOMALY SHIELD</div>
    <div class="hero-line"></div>
    <div class="hero-sub">Обнаружение аномалий в показаниях IoT-датчиков умного города</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Инициализация модели..."):
    rf_model, scaler, feature_cols = train_model()

st.markdown('<div class="status-ok">✅ Модель готова | Random Forest 300 деревьев | Пороги: T>50°C, Battery<10%</div>', unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">▶ ЗАГРУЗКА ДАННЫХ</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("ZIP или CSV", type=["zip","csv"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem 1rem;color:#1a3a4a;">
        <div style="font-size:3.5rem;margin-bottom:1rem;">📡</div>
        <div style="font-family:Orbitron,monospace;font-size:1rem;color:#00ffb4;letter-spacing:0.1em;">ОЖИДАНИЕ ДАННЫХ</div>
        <div style="margin-top:0.8rem;font-size:0.75rem;color:#2a4a5a;">
            Ожидаемые колонки: <b>Device_ID · Temperature · Humidity · Battery_Level</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_raw, fname = load_data(uploaded)
if df_raw is None:
    st.error(fname); st.stop()

st.markdown(f'<div class="status-ok">📄 {fname} · {len(df_raw):,} записей · устройств: {df_raw["Device_ID"].nunique() if "Device_ID" in df_raw.columns else "?"}</div>', unsafe_allow_html=True)

df, X_scaled, err = preprocess(df_raw.copy(), scaler, feature_cols)
if err:
    st.error(err); st.stop()

proba    = rf_model.predict_proba(X_scaled)
preds    = rf_model.predict(X_scaled)
max_conf = proba.max(axis=1) * 100

df['Предсказание']    = preds
df['Уверенность (%)'] = max_conf.round(2)
df['Тип аномалии']    = df.apply(lambda r: anomaly_type(r, r['Предсказание']), axis=1)
df['P(Normal)%']      = (proba[:, 0] * 100).round(1)
df['P(Anomaly)%']     = (proba[:, 1] * 100).round(1)
if show_reasons:
    df['Причина'] = df.apply(lambda r: explain_anomaly(r, r['Предсказание']), axis=1)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">▶ СВОДКА</div>', unsafe_allow_html=True)

total  = len(df)
n_norm = (df['Предсказание'] == 0).sum()
n_anom = (df['Предсказание'] == 1).sum()
n_heat = df['Тип аномалии'].str.contains('Перегрев').sum()
n_batt = df['Тип аномалии'].str.contains('Разряд').sum()
pct    = n_anom / total * 100 if total else 0

c1,c2,c3,c4,c5 = st.columns(5)
for col, cls, ncls, val, lbl in [
    (c1,'card-total',  'num-total',   total,   'ВСЕГО'),
    (c2,'card-normal', 'num-normal',  n_norm,  'НОРМА'),
    (c3,'card-anomaly','num-anomaly', n_anom,  'АНОМАЛИЙ'),
    (c4,'card-heat',   'num-heat',    n_heat,  'ПЕРЕГРЕВ'),
    (c5,'card-battery','num-battery', n_batt,  'РАЗРЯД'),
]:
    with col:
        st.markdown(f'<div class="metric-card {cls}"><div class="metric-num {ncls}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">▶ ВИЗУАЛИЗАЦИЯ</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Распределение классов**")
    counts = df['Тип аномалии'].value_counts()
    colors_pie = [COLORS.get(l,'#888') for l in counts.index]
    fig,ax = plt.subplots(figsize=(4,4),facecolor='none')
    ax.pie(counts.values, labels=None, colors=colors_pie, autopct='%1.1f%%',
           startangle=140, wedgeprops={'linewidth':2,'edgecolor':'#040810'}, pctdistance=0.75)
    patches = [mpatches.Patch(color=COLORS.get(l,'#888'),label=l) for l in counts.index]
    ax.legend(handles=patches,loc='lower center',bbox_to_anchor=(0.5,-0.12),ncol=1,
              frameon=False,labelcolor='#8b9ab0',fontsize=7)
    fig.patch.set_alpha(0)
    st.pyplot(fig); plt.close()

with col_b:
    st.markdown("**Важность признаков**")
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    fig,ax = dark_fig(5,4)
    colors_bar = ['#00ffb4' if v > imp.median() else '#1a3a4a' for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors_bar, height=0.6, edgecolor='none')
    ax.set_xlabel('Важность', color='#5a7a8a', fontsize=8)
    fig.patch.set_alpha(0); plt.tight_layout()
    st.pyplot(fig); plt.close()

with col_c:
    st.markdown("**Температура vs Батарея**")
    fig,ax = dark_fig(5,4)
    norm_df = df[df['Предсказание']==0]
    anom_df = df[df['Предсказание']==1]
    ax.scatter(norm_df['Temperature'], norm_df['Battery_Level'], c='#00ffb4', alpha=0.4, s=10, label='Норма')
    ax.scatter(anom_df['Temperature'], anom_df['Battery_Level'], c='#ff4444', alpha=0.7, s=15, label='Аномалия')
    ax.axvline(x=THRESHOLDS['temp_anomaly_min'], color='#ff8c00', linestyle='--', linewidth=1, alpha=0.7, label=f"T>50°C")
    ax.axhline(y=THRESHOLDS['battery_critical'], color='#facc15', linestyle='--', linewidth=1, alpha=0.7, label=f"B<10%")
    ax.set_xlabel('Температура (°C)', color='#5a7a8a', fontsize=8)
    ax.set_ylabel('Заряд батареи (%)', color='#5a7a8a', fontsize=8)
    ax.legend(frameon=False, labelcolor='#8b9ab0', fontsize=7)
    fig.patch.set_alpha(0); plt.tight_layout()
    st.pyplot(fig); plt.close()

if show_timeline and 'Device_ID' in df.columns:
    st.markdown("**Аномалии по устройствам**")
    device_stats = df.groupby('Device_ID').agg(
        Total=('Предсказание','count'), Anomalies=('Предсказание','sum')
    ).reset_index()
    device_stats['Anomaly_Rate'] = device_stats['Anomalies'] / device_stats['Total'] * 100
    device_stats = device_stats.sort_values('Anomaly_Rate', ascending=False).head(20)
    fig,ax = dark_fig(10,3)
    colors_dev = ['#ff4444' if r>50 else '#ff8c00' if r>20 else '#00ffb4' for r in device_stats['Anomaly_Rate']]
    ax.bar(device_stats['Device_ID'], device_stats['Anomaly_Rate'], color=colors_dev, edgecolor='none')
    ax.set_ylabel('% аномалий', color='#5a7a8a', fontsize=8)
    ax.set_title('Топ устройств по доле аномалий', color='#5a7a8a', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    fig.patch.set_alpha(0); plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── ANOMALY TABLE ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">▶ ОБНАРУЖЕННЫЕ АНОМАЛИИ</div>', unsafe_allow_html=True)
anomalies = df[df['Предсказание'] == 1].copy()

if len(anomalies) == 0:
    st.markdown('<div class="status-ok">✅ Аномалий не обнаружено</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-warn">⚠ ОБНАРУЖЕНО {len(anomalies):,} АНОМАЛИЙ ({pct:.1f}% устройств)</div>', unsafe_allow_html=True)

    display_cols = []
    if 'Device_ID' in anomalies.columns: display_cols.append('Device_ID')
    display_cols += ['Temperature','Humidity','Battery_Level','Тип аномалии','Уверенность (%)','P(Normal)%','P(Anomaly)%']
    if show_reasons: display_cols.append('Причина')
    existing = [c for c in display_cols if c in anomalies.columns]

    def highlight(row):
        t = row.get('Тип аномалии','')
        if 'Перегрев + Разряд' in t: bg='#2a0a00'
        elif 'Перегрев' in t: bg='#2a0800'
        elif 'Разряд' in t: bg='#2a2000'
        else: bg='#1a0a2a'
        return [f'background-color:{bg}']*len(row)

    styled = (anomalies[existing].reset_index(drop=True)
              .style.apply(highlight, axis=1)
              .format({'Уверенность (%)':'{:.1f}%','Temperature':'{:.1f}°C',
                       'Humidity':'{:.1f}%','P(Normal)%':'{:.1f}%','P(Anomaly)%':'{:.1f}%'}))
    st.dataframe(styled, use_container_width=True, height=420)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("⬇ Скачать аномалии (CSV)",
                           anomalies[existing].to_csv(index=False).encode('utf-8'),
                           "anomalies.csv", "text/csv")
    with col_d2:
        heat = anomalies[anomalies['Тип аномалии'].str.contains('Перегрев')]
        if len(heat):
            st.download_button("⬇ Только перегрев (CSV)",
                               heat[existing].to_csv(index=False).encode('utf-8'),
                               "overheating.csv","text/csv")

with st.expander("📋 Полная таблица"):
    fc = [c for c in ['Device_ID','Temperature','Humidity','Battery_Level','Тип аномалии','Уверенность (%)'] if c in df.columns]
    st.dataframe(df[fc], use_container_width=True, height=350)

st.markdown('<div style="text-align:center;padding:2rem 0 1rem;color:#1a3a4a;font-size:0.65rem;border-top:1px solid #0a1a2a;margin-top:2rem;">IoT Anomaly Shield · Дипломная работа · Astana IT University</div>', unsafe_allow_html=True)
