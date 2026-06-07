import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

st.set_page_config(page_title="IoT Threat Shield", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Orbitron:wght@400;700;900&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;background:#040810;color:#c9d1d9;}
.stApp{background:#040810;}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;
  background-image:linear-gradient(rgba(0,255,180,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,180,0.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0;}
.hero{text-align:center;padding:2rem 1rem 1rem;}
.hero-title{font-family:'Orbitron',monospace;font-weight:900;font-size:2.8rem;color:#00ffb4;text-shadow:0 0 30px rgba(0,255,180,0.5);letter-spacing:0.08em;}
.hero-sub{font-size:0.72rem;letter-spacing:0.25em;color:#3d5a6e;text-transform:uppercase;}
.hero-line{width:200px;height:1px;background:linear-gradient(90deg,transparent,#00ffb4,transparent);margin:0.8rem auto;}
.section-header{font-family:'Orbitron',monospace;font-size:0.85rem;letter-spacing:0.2em;text-transform:uppercase;color:#00ffb4;border-bottom:1px solid rgba(0,255,180,0.15);padding-bottom:0.5rem;margin:2rem 0 1rem;}
.status-bar{font-size:0.72rem;color:#00ffb4;background:rgba(0,255,180,0.06);border-left:3px solid #00ffb4;padding:0.5rem 1rem;border-radius:0 6px 6px 0;margin:0.8rem 0;}
.status-warn{color:#ff4444;background:rgba(255,68,68,0.06);border-left-color:#ff4444;}
.metric-card{background:rgba(0,255,180,0.03);border:1px solid rgba(0,255,180,0.12);border-radius:8px;padding:1.2rem;text-align:center;position:relative;overflow:hidden;}
.metric-num{font-family:'Orbitron',monospace;font-size:2rem;font-weight:700;line-height:1;}
.metric-label{font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:#3d5a6e;margin-top:0.3rem;}
.num-normal{color:#00ffb4;} .num-ddos{color:#ff4444;} .num-total{color:#4488ff;}
.rule-card{background:rgba(255,255,255,0.02);border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;}
.rule-normal{border-left:3px solid #00ffb4;} .rule-ddos{border-left:3px solid #ff4444;} .rule-portscan{border-left:3px solid #ff8c00;}
.rule-title{font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;}
.rule-param{font-size:0.68rem;color:#8b9ab0;line-height:1.7;}
.rule-val{color:#e2e8f0;font-weight:700;}
div[data-testid="stFileUploader"]{background:rgba(0,255,180,0.03);border:2px dashed rgba(0,255,180,0.2);border-radius:10px;padding:1rem;}
div[data-testid="stSidebar"]{background:#02060f;border-right:1px solid rgba(0,255,180,0.08);}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IOT_THRESHOLDS = {
    'ddos':     {'Packets': 5000, 'Bytes': 100000, 'Packet_Rate': 1000, 'Byte_Rate': 50000},
    'portscan': {'Flow_Duration': 0.2, 'Packets': 4, 'Packet_Size_Avg': 60}
}
PROTOCOL_MAP = {0:'HOPOPT', 1:'ICMP', 6:'TCP', 17:'UDP', 58:'ICMPv6'}
COLORS = {'Normal':'#00ffb4','DDoS':'#ff4444','PortScan':'#ff8c00','Anomaly':'#ff4444','Неопределено':'#4488ff'}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_name(s): return str(s).strip().replace(' ','_').replace('-','_')

def load_data(f):
    if f.name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(f.read())) as z:
            csvs = [x for x in z.namelist() if x.lower().endswith('.csv')]
            if not csvs: return None, 'ZIP не содержит CSV'
            with z.open(csvs[0]) as c: return pd.read_csv(c), csvs[0]
    return pd.read_csv(f), f.name

def find_label_col(df):
    for c in df.columns:
        if c.lower() in ['label','target','class','attack','category','type','anomaly']: return c
    return None

def first_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in names): return c
    return None

def normalize_protocol(df):
    for c in df.columns:
        if c.lower() in ['protocol','proto']:
            s = df[c]
            if s.dtype == object:
                return s.str.upper().map({'TCP':6,'UDP':17,'ICMP':1,'ICMPV6':58}).fillna(0)
            return pd.to_numeric(s, errors='coerce').fillna(0)
    proto_cols = [c for c in df.columns if 'protocol' in c.lower()]
    out = pd.Series(0, index=df.index, dtype=float)
    for c in proto_cols:
        val = pd.to_numeric(df[c], errors='coerce').fillna(0)
        if 'TCP' in c.upper(): out = np.where(val>0, 6, out)
        elif 'UDP' in c.upper(): out = np.where(val>0, 17, out)
        elif 'ICMP' in c.upper(): out = np.where(val>0, 1, out)
    return pd.Series(out, index=df.index)

def add_iot_features(df):
    df = df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.select_dtypes(include=['bool']).columns: df[c] = df[c].astype(int)
    eps = 1e-9
    pkt = first_col(df, ['Packets','packet_count','packet_count_5s','total_packets'])
    byt = first_col(df, ['Bytes','bytes','total_bytes','flow_bytes','packet_size_total'])
    dur = first_col(df, ['Flow_Duration','duration','flow_duration','inter_arrival_time'])
    siz = first_col(df, ['Packet_Size_Avg','mean_packet_size','packet_size','packet_len'])
    if 'Packets' not in df.columns:
        df['Packets'] = pd.to_numeric(df[pkt], errors='coerce') if pkt else 1
    if 'Flow_Duration' not in df.columns:
        df['Flow_Duration'] = pd.to_numeric(df[dur], errors='coerce') if dur else 1.0
    if 'Packet_Size_Avg' not in df.columns:
        df['Packet_Size_Avg'] = pd.to_numeric(df[siz], errors='coerce') if siz else np.nan
    if 'Bytes' not in df.columns:
        df['Bytes'] = pd.to_numeric(df[byt], errors='coerce') if byt else df['Packets'] * df['Packet_Size_Avg'].fillna(100)
    if 'Protocol' not in df.columns:
        df['Protocol'] = normalize_protocol(df)
    for c in ['Packets','Bytes','Flow_Duration','Packet_Size_Avg','Protocol']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Packets'] = df['Packets'].fillna(df['Packets'].median()).fillna(1)
    df['Bytes'] = df['Bytes'].fillna(df['Bytes'].median()).fillna(100)
    df['Flow_Duration'] = df['Flow_Duration'].fillna(df['Flow_Duration'].median()).fillna(1)
    df['Packet_Size_Avg'] = df['Packet_Size_Avg'].fillna(df['Bytes']/(df['Packets']+eps)).fillna(100)
    df['Protocol'] = df['Protocol'].fillna(0)
    df['Packet_Rate'] = df['Packets'] / (df['Flow_Duration'].abs() + eps)
    df['Byte_Rate']   = df['Bytes']   / (df['Flow_Duration'].abs() + eps)
    return df

def make_X(df, label_col=None):
    X = df.drop(columns=[label_col], errors='ignore').copy()
    for c in X.select_dtypes(include=['bool']).columns: X[c] = X[c].astype(int)
    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X

def normalize_label(x):
    s = str(x).strip().lower()
    if s in ['0','0.0','normal','benign','legitimate']: return 'Normal'
    if 'ddos' in s or 'dos' in s: return 'DDoS'
    if 'port' in s or 'scan' in s: return 'PortScan'
    if s in ['1','1.0','attack','anomaly','malicious','abnormal']: return 'Anomaly'
    return str(x).strip()

def get_protocol_name(val):
    try: return PROTOCOL_MAP.get(int(val), f'Proto-{int(val)}')
    except: return str(val)

def explain_anomaly(row, label):
    reasons = []
    t = IOT_THRESHOLDS
    if row.get('Packets',0) > t['ddos']['Packets']:
        reasons.append(f"Packets={row['Packets']:.0f} — слишком много пакетов для IoT")
    if row.get('Bytes',0) > t['ddos']['Bytes']:
        reasons.append(f"Bytes={row['Bytes']:.0f} — слишком большой объём трафика")
    if row.get('Packet_Rate',0) > t['ddos']['Packet_Rate']:
        reasons.append(f"Packet_Rate={row['Packet_Rate']:.1f}/с — высокая частота пакетов")
    if row.get('Byte_Rate',0) > t['ddos']['Byte_Rate']:
        reasons.append(f"Byte_Rate={row['Byte_Rate']:.1f} Б/с — высокий поток байтов")
    if row.get('Flow_Duration',99) < t['portscan']['Flow_Duration'] and row.get('Packets',99) <= t['portscan']['Packets']:
        reasons.append("Очень короткий поток + мало пакетов — похоже на PortScan")
    if row.get('Packet_Size_Avg',999) < t['portscan']['Packet_Size_Avg']:
        reasons.append(f"Packet_Size_Avg={row['Packet_Size_Avg']:.1f} — маленькие пакеты (только заголовки)")
    proto = get_protocol_name(row.get('Protocol',-1))
    if 'PortScan' in str(label) and proto == 'TCP':
        reasons.append('TCP — типичен для SYN/port scanning')
    for c in row.index:
        try:
            v = float(row[c])
            if v > 0 and any(k in c.lower() for k in ['syn','fin','entropy','frequency']):
                reasons.append(f"{c}={row[c]} — аномальный сетевой признак")
        except: pass
    if not reasons: reasons.append('Аномалия по комбинации признаков модели')
    return ' | '.join(reasons[:4])

def train_from_uploaded(df, label_col):
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    if len(np.unique(y)) < 2:
        return None, None, None, None, 'В label только один класс'
    X = make_X(df, label_col)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # balance without SMOTE
    df_bal = pd.DataFrame(X_scaled, columns=X.columns)
    df_bal['__y__'] = y
    maj = df_bal[df_bal['__y__'] == df_bal['__y__'].mode()[0]]
    mins = [df_bal[df_bal['__y__'] == c] for c in np.unique(y) if c != df_bal['__y__'].mode()[0]]
    balanced = pd.concat([maj] + [resample(m, replace=True, n_samples=len(maj), random_state=42) for m in mins])
    y_bal = balanced['__y__'].values
    X_bal = balanced.drop(columns=['__y__']).values
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    return rf, scaler, le, list(X.columns), None

@st.cache_resource(show_spinner=False)
def train_fallback():
    np.random.seed(42); n = 6000
    norm = pd.DataFrame({'Flow_Duration':np.random.exponential(.5,n//3).clip(.001,5),'Packets':np.random.randint(1,100,n//3),'Bytes':np.random.randint(40,2000,n//3),'Protocol':np.random.choice([6,17,1],n//3),'Label':'Normal'})
    ddos = pd.DataFrame({'Flow_Duration':np.random.uniform(1,60,n//3),'Packets':np.random.randint(5000,80000,n//3),'Bytes':np.random.randint(100000,5000000,n//3),'Protocol':np.random.choice([6,17,1],n//3),'Label':'DDoS'})
    ps   = pd.DataFrame({'Flow_Duration':np.random.uniform(.0001,.15,n//3),'Packets':np.random.randint(1,4,n//3),'Bytes':np.random.randint(40,120,n//3),'Protocol':np.random.choice([6,17],n//3,p=[.85,.15]),'Label':'PortScan'})
    d = add_iot_features(pd.concat([norm,ddos,ps],ignore_index=True))
    le = LabelEncoder(); y = le.fit_transform(d['Label'])
    feats = ['Flow_Duration','Packets','Bytes','Protocol','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    sc = StandardScaler(); X = sc.fit_transform(d[feats])
    rf = RandomForestClassifier(n_estimators=300,max_depth=20,class_weight='balanced',random_state=42,n_jobs=-1).fit(X,y)
    return rf, sc, le, feats

def dark_fig(w,h):
    fig,ax=plt.subplots(figsize=(w,h),facecolor='none'); ax.set_facecolor('#040810')
    for s in ax.spines.values(): s.set_color('#1a2535')
    ax.tick_params(colors='#5a7a8a',labelsize=9); return fig,ax

def pie_chart(counts):
    fig,ax=plt.subplots(figsize=(4.5,4.5),facecolor='none')
    colors=[COLORS.get(l,'#888') for l in counts.index]
    ax.pie(counts.values,colors=colors,autopct='%1.1f%%',startangle=140,wedgeprops={'linewidth':2,'edgecolor':'#040810'})
    ax.legend(handles=[mpatches.Patch(color=COLORS.get(l,'#888'),label=l) for l in counts.index],
              loc='lower center',bbox_to_anchor=(0.5,-.08),ncol=2,frameon=False,labelcolor='#8b9ab0',fontsize=8)
    fig.patch.set_alpha(0); return fig

def bar_chart(importances,cols):
    fig,ax=dark_fig(7,3.2)
    imp=pd.Series(importances,index=cols).sort_values().tail(15)
    ax.barh(imp.index,imp.values,color='#00ffb4',edgecolor='none')
    ax.set_xlabel('Важность',color='#5a7a8a',fontsize=8); fig.patch.set_alpha(0); plt.tight_layout(); return fig

def hist_chart(df):
    fig,ax=dark_fig(7,3.2)
    for lbl in df['Статус'].unique():
        ax.hist(df[df['Статус']==lbl]['Уверенность (%)'],bins=20,alpha=.7,label=lbl,color=COLORS.get(lbl,'#4488ff'))
    ax.set_xlabel('Уверенность (%)',color='#5a7a8a'); ax.legend(frameon=False,labelcolor='#8b9ab0',fontsize=8)
    fig.patch.set_alpha(0); plt.tight_layout(); return fig

def timeline_chart(df):
    fig,ax=dark_fig(10,2.5)
    colors=[COLORS.get(l,'#4488ff') for l in df['Статус']]
    ax.bar(np.arange(len(df)),1,color=colors,width=1,edgecolor='none')
    ax.set_yticks([]); ax.set_xlabel('Индекс записи',color='#5a7a8a',fontsize=8)
    ax.set_title('Timeline трафика',color='#5a7a8a',fontsize=9)
    patches=[mpatches.Patch(color=COLORS.get(l,'#888'),label=l) for l in df['Статус'].unique()]
    ax.legend(handles=patches,frameon=False,labelcolor='#8b9ab0',fontsize=8,loc='upper right')
    fig.patch.set_alpha(0); plt.tight_layout(); return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.8rem;color:#00ffb4;letter-spacing:0.2em;margin-bottom:1rem;">⚙ ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider('Порог уверенности (%)', 50, 99, 65)
    show_reasons  = st.checkbox('Показывать причины аномалий', value=True)
    show_proba    = st.checkbox('Вероятности классов', value=True)
    show_timeline = st.checkbox('Timeline трафика', value=True)
    st.markdown('---')
    st.markdown("""
    <div style="font-family:Orbitron,monospace;font-size:0.72rem;color:#00ffb4;letter-spacing:0.15em;margin-bottom:0.8rem;">📡 IoT ПОРОГИ ДЕТЕКЦИИ</div>
    <div class="rule-card rule-normal"><div class="rule-title">✅ Нормальный IoT трафик</div>
    <div class="rule-param">Пакеты: <span class="rule-val">1–100</span><br>Байты: <span class="rule-val">40–2,000</span><br>Длит. потока: <span class="rule-val">0.001–5.0 с</span><br>Протоколы: <span class="rule-val">TCP / UDP / ICMP</span></div></div>
    <div class="rule-card rule-ddos"><div class="rule-title">🔴 DDoS атака</div>
    <div class="rule-param">Пакеты: <span class="rule-val">&gt;5,000</span> (флуд)<br>Байты: <span class="rule-val">&gt;100,000</span><br>Packet Rate: <span class="rule-val">&gt;1,000/с</span><br>Byte Rate: <span class="rule-val">&gt;50,000 Б/с</span></div></div>
    <div class="rule-card rule-portscan"><div class="rule-title">🟠 PortScan</div>
    <div class="rule-param">Короткий поток <span class="rule-val">&lt;0.2с</span><br>Пакеты: <span class="rule-val">1–3</span> (SYN)<br>Avg размер: <span class="rule-val">&lt;60 Б</span><br>Протокол: <span class="rule-val">TCP</span></div></div>
    """, unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div style="font-size:0.65rem;color:#3d5a6e;">Модель: Random Forest 300 деревьев<br>Адаптивный режим: любой CSV/ZIP<br>Авто-определение колонок</div>', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡 IoT THREAT SHIELD</div>
    <div class="hero-line"></div>
    <div class="hero-sub">Система обнаружения аномалий в сетевом трафике IoT-устройств</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="status-bar">✅ Адаптивный режим | ZIP/CSV | авто-колонки | причины аномалий сохранены</div>', unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ЗАГРУЗКА ДАТАСЕТА</div>', unsafe_allow_html=True)
uploaded = st.file_uploader('Перетащите ZIP-архив или CSV-файл', type=['zip','csv'], label_visibility='collapsed')

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem 1rem;color:#1a3a4a;">
        <div style="font-size:3.5rem;">📡</div>
        <div style="font-family:Orbitron,monospace;font-size:1rem;color:#00ffb4;letter-spacing:0.1em;">ОЖИДАНИЕ ДАННЫХ</div>
        <div style="margin-top:0.8rem;font-size:0.75rem;color:#2a4a5a;">Загрузите ZIP или CSV. Колонки будут адаптированы автоматически.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df_raw, fname = load_data(uploaded)
if df_raw is None: st.error(fname); st.stop()

st.markdown(f'<div class="status-bar">📄 {fname} · {len(df_raw):,} записей · {len(df_raw.columns)} колонок</div>', unsafe_allow_html=True)

df = add_iot_features(df_raw)
label_col = find_label_col(df)

if label_col:
    rf_model, scaler, le, feature_cols, err = train_from_uploaded(df, label_col)
    if err: st.error(err); st.stop()
    X = make_X(df, label_col).reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(X)
    model_mode = f'модель обучена на загруженном датасете по колонке `{label_col}`'
else:
    rf_model, scaler, le, feature_cols = train_fallback()
    X_scaled = scaler.transform(df[feature_cols])
    model_mode = 'label не найден — используется встроенная IoT-модель'

st.markdown(f'<div class="status-bar">✅ {model_mode}</div>', unsafe_allow_html=True)

proba     = rf_model.predict_proba(X_scaled)
preds     = rf_model.predict(X_scaled)
pred_lbls = [normalize_label(x) for x in le.inverse_transform(preds)]
max_conf  = proba.max(axis=1) * 100

df['Предсказание']    = pred_lbls
df['Уверенность (%)'] = max_conf.round(2)
df['Статус']          = np.where(max_conf >= confidence_threshold, df['Предсказание'], 'Неопределено')
for i, cls in enumerate(le.classes_):
    df[f'P({normalize_label(cls)})%'] = (proba[:,i]*100).round(1)
df['Протокол'] = df['Protocol'].apply(get_protocol_name)
if show_reasons:
    df['Причина'] = df.apply(lambda r: explain_anomaly(r, r['Статус']) if r['Статус'] not in ['Normal','Неопределено'] else '', axis=1)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ СВОДКА АНАЛИЗА</div>', unsafe_allow_html=True)
total   = len(df)
n_norm  = (df['Статус']=='Normal').sum()
n_undef = (df['Статус']=='Неопределено').sum()
n_threat= total - n_norm - n_undef
pct     = n_threat/total*100 if total else 0

cols = st.columns(5)
for col,(val,lab,cls) in zip(cols,[
    (total,'ВСЕГО ЗАПИСЕЙ','num-total'),
    (n_norm,'НОРМАЛЬНЫЙ','num-normal'),
    (n_threat,'АНОМАЛИИ','num-ddos'),
    (n_undef,'НЕОПРЕДЕЛЕНО','num-total'),
    (f'{pct:.1f}%','УГРОЗ','num-ddos')
]):
    with col: st.markdown(f'<div class="metric-card"><div class="metric-num {cls}">{val}</div><div class="metric-label">{lab}</div></div>', unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ВИЗУАЛИЗАЦИЯ</div>', unsafe_allow_html=True)
c1,c2,c3 = st.columns([1.2,1.8,1.8])
with c1:
    st.markdown('**Классификация трафика**')
    st.pyplot(pie_chart(df['Статус'].value_counts()))
with c2:
    st.markdown('**Важность признаков (RF)**')
    st.pyplot(bar_chart(rf_model.feature_importances_, feature_cols))
with c3:
    st.markdown('**Распределение уверенности**')
    st.pyplot(hist_chart(df))

if show_timeline:
    st.markdown('**Timeline трафика**')
    st.pyplot(timeline_chart(df))

# ── ANOMALY TABLE ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">▶ ОБНАРУЖЕННЫЕ УГРОЗЫ</div>', unsafe_allow_html=True)
anomalies = df[~df['Статус'].isin(['Normal','Неопределено'])].copy()

if len(anomalies) == 0:
    st.markdown('<div class="status-bar">✅ Угроз не обнаружено — трафик в норме</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-bar status-warn">⚠ ОБНАРУЖЕНО {len(anomalies):,} АНОМАЛЬНЫХ ПОТОКОВ ({pct:.1f}% трафика)</div>', unsafe_allow_html=True)
    display_cols = ['Статус','Уверенность (%)','Протокол','Flow_Duration','Packets','Bytes','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    if show_proba:    display_cols += [c for c in df.columns if c.startswith('P(')]
    if show_reasons:  display_cols.append('Причина')
    display_cols = [c for c in display_cols if c in anomalies.columns]

    def highlight(row):
        s = row.get('Статус','')
        bg = '#2a0808' if 'DDoS' in s or 'Anomaly' in s else '#2a1400' if 'PortScan' in s else ''
        return [f'background-color:{bg}']*len(row)

    styled = anomalies[display_cols].reset_index(drop=True).style.apply(highlight, axis=1).format({'Уверенность (%)':'{:.1f}%'})
    st.dataframe(styled, use_container_width=True, height=420)
    st.download_button('⬇ Скачать аномалии (CSV)', anomalies[display_cols].to_csv(index=False).encode('utf-8'), 'anomalies.csv', 'text/csv')

with st.expander('📋 Полная таблица всех записей'):
    fc = ['Статус','Уверенность (%)','Протокол','Flow_Duration','Packets','Bytes','Packet_Rate','Byte_Rate','Packet_Size_Avg']
    if show_reasons: fc.append('Причина')
    st.dataframe(df[[c for c in fc if c in df.columns]], use_container_width=True, height=350)

st.markdown('<div style="text-align:center;padding:2rem 0 1rem;color:#1a3a4a;font-size:0.65rem;border-top:1px solid #0a1a2a;margin-top:2rem;">IoT Threat Shield · Дипломная работа · Astana IT University</div>', unsafe_allow_html=True)
