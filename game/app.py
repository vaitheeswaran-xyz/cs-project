"""
STARFALL: A Data Science Adventure
----------------------------------
You are the Data Officer of the starship Meridian. Every mission teaches one
core data-science idea through hands-on play:

    1. Signal Scrub      -> data cleaning (NaNs, outliers, scaling)
    2. Threat Scanner    -> classification (threshold, precision vs recall)
    3. Orbit Tuner       -> regression (under/overfitting, regularization)
    4. Fleet Finder      -> clustering (choosing K, silhouette)

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_curve, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

# ==========================================================
# PAGE CONFIG & THEME
# ==========================================================
st.set_page_config(
    page_title="STARFALL | Data Science Adventure",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,0.22), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,0.14), transparent 60%),
        #060912;
    color: #e5e7f0;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }

h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.5px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #080b17 100%);
    border-right: 1px solid rgba(148,163,184,0.12);
}
.logo {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 1.5rem;
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub { color: #94a3b8; font-size: 0.78rem; margin-top: -4px; margin-bottom: 14px; }

/* Glass cards */
.glass {
    background: rgba(17, 24, 39, 0.62);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 18px; padding: 18px 22px; margin-bottom: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}
.hero {
    background: linear-gradient(135deg, rgba(34,211,238,0.14), rgba(139,92,246,0.18));
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 22px; padding: 28px 32px; margin-bottom: 18px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.2rem; }
.chapter {
    font-family: 'Orbitron', sans-serif; font-size: 0.75rem; letter-spacing: 3px;
    color: #22d3ee; text-transform: uppercase;
}
.title-grad {
    background: linear-gradient(90deg, #e0f2fe, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* HUD */
.hud { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.hud-item {
    flex: 1; min-width: 150px; padding: 10px 16px; border-radius: 14px;
    background: rgba(15,23,42,0.75); border: 1px solid rgba(148,163,184,0.16);
}
.hud-label { font-size: 0.68rem; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase; }
.hud-value { font-family: 'Orbitron', sans-serif; font-size: 1.15rem; color: #f8fafc; }
.bar { height: 8px; border-radius: 99px; background: rgba(148,163,184,0.18); overflow: hidden; margin-top: 6px; }
.bar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22d3ee, #a78bfa); transition: width .6s ease; }

/* NOVA dialogue */
.nova { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 16px; }
.nova-avatar {
    flex: 0 0 52px; height: 52px; border-radius: 50%; display: grid; place-items: center;
    font-size: 1.6rem; background: radial-gradient(circle at 30% 30%, #67e8f9, #6366f1);
    box-shadow: 0 0 22px rgba(99,102,241,0.7); animation: pulse 2.6s infinite;
}
.nova-bubble {
    background: rgba(30,41,59,0.75); border: 1px solid rgba(103,232,249,0.3);
    border-radius: 4px 18px 18px 18px; padding: 12px 18px; line-height: 1.55;
}
.nova-name { font-family: 'Orbitron', sans-serif; font-size: 0.7rem; color: #67e8f9; letter-spacing: 2px; margin-bottom: 2px; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 14px rgba(99,102,241,0.6);} 50% { box-shadow: 0 0 30px rgba(103,232,249,0.9);} }

/* Objectives */
.obj { padding: 8px 12px; border-radius: 10px; margin: 6px 0; background: rgba(15,23,42,0.6);
       border: 1px solid rgba(148,163,184,0.14); font-size: 0.93rem; }
.obj.ok { border-color: rgba(52,211,153,0.6); background: rgba(6,78,59,0.35); }

/* Stat tiles */
.tiles { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.tile { padding: 12px; border-radius: 14px; text-align: center; border: 1px solid rgba(148,163,184,0.18); background: rgba(15,23,42,0.7); }
.tile b { display: block; font-family: 'Orbitron', sans-serif; font-size: 1.6rem; }
.tile span { font-size: 0.75rem; color: #94a3b8; }
.tile.good b { color: #34d399; } .tile.bad b { color: #f87171; } .tile.warn b { color: #fbbf24; }

/* Victory */
.victory {
    text-align: center; padding: 18px; border-radius: 18px; margin: 10px 0;
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(34,211,238,0.2));
    border: 1px solid rgba(52,211,153,0.7); box-shadow: 0 0 30px rgba(52,211,153,0.25);
}
.victory h3 { margin: 0; color: #6ee7b7; }

/* Buttons */
.stButton > button {
    border-radius: 12px; border: 1px solid rgba(167,139,250,0.5); font-weight: 600;
    background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; padding: 0.55rem 1.1rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 22px rgba(124,58,237,0.55); border-color: #c4b5fd; color: #fff; }
.stButton > button:disabled { opacity: 0.4; }

[data-testid="stMetric"] {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.16);
    padding: 10px 14px; border-radius: 14px;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.locked { text-align: center; padding: 60px 20px; }
.locked .big { font-size: 4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==========================================================
# GAME STATE
# ==========================================================
MISSION_REWARD = 100
HINT_PENALTY = 20
RANKS = ["Cadet", "Navigator", "Analyst", "Specialist", "Commander"]

STAGES = {
    0: ("🛰️", "Prologue: Wake Up"),
    1: ("🧹", "Mission 1: Signal Scrub"),
    2: ("🎯", "Mission 2: Threat Scanner"),
    3: ("🌀", "Mission 3: Orbit Tuner"),
    4: ("🛸", "Mission 4: Fleet Finder"),
    5: ("🏆", "Debrief: Mission Log"),
}

if "xp" not in st.session_state:
    st.session_state.xp = 0
    st.session_state.done = set()
    st.session_state.hints = set()
    st.session_state.nav = 0


def is_unlocked(stage: int) -> bool:
    """Return True if the player may open `stage` (missions unlock in order; the debrief needs all four)."""
    if stage <= 1:
        return True
    if stage == 5:
        return {1, 2, 3, 4} <= st.session_state.done
    return (stage - 1) in st.session_state.done


def set_tau(v: float):
    st.session_state.tau = v


def go_to(stage: int):
    st.session_state.nav = stage


def reset_game():
    st.session_state.xp = 0
    st.session_state.done = set()
    st.session_state.hints = set()
    st.session_state.nav = 0


def claim(stage: int):
    """Bank the XP reward for a finished mission (reduced if a hint was used)."""
    reward = MISSION_REWARD - (HINT_PENALTY if stage in st.session_state.hints else 0)
    st.session_state.xp += reward
    st.session_state.done.add(stage)
    st.toast(f"+{reward} XP: mission complete!", icon="⭐")


def rank_info():
    """Return (current rank name, XP needed for next rank, progress % toward it)."""
    xp = st.session_state.xp
    idx = min(xp // MISSION_REWARD, len(RANKS) - 1)
    nxt = (idx + 1) * MISSION_REWARD
    pct = 100 if idx == len(RANKS) - 1 else int((xp - idx * MISSION_REWARD) / MISSION_REWARD * 100)
    return RANKS[idx], nxt, pct


# ==========================================================
# UI HELPERS
# ==========================================================
def html(s: str):
    """Render raw HTML. Newlines are stripped so Markdown doesn't break the markup."""
    st.markdown(s.replace("\n", " "), unsafe_allow_html=True)


def hud():
    """Top status bar: rank, total XP and ship-repair progress."""
    rank, _, pct = rank_info()
    done = len([d for d in st.session_state.done if 1 <= d <= 4])
    html(
        f"<div class='hud'>"
        f"<div class='hud-item'><div class='hud-label'>Rank</div><div class='hud-value'>{rank}</div>"
        f"<div class='bar'><div style='width:{pct}%'></div></div></div>"
        f"<div class='hud-item'><div class='hud-label'>Experience</div><div class='hud-value'>{st.session_state.xp} XP</div></div>"
        f"<div class='hud-item'><div class='hud-label'>Ship Repair</div><div class='hud-value'>{done}/4 systems</div>"
        f"<div class='bar'><div style='width:{done * 25}%'></div></div></div>"
        f"</div>"
    )


def mission_header(chapter: str, title: str):
    html(f"<div class='chapter'>{chapter}</div><h1 class='title-grad' style='margin-top:0'>{title}</h1>")


def nova(text: str):
    """Show a dialogue bubble from NOVA, the ship AI who narrates each mission."""
    html(
        f"<div class='nova'><div class='nova-avatar'>🤖</div>"
        f"<div class='nova-bubble'><div class='nova-name'>NOVA · SHIP AI</div>{text}</div></div>"
    )


def objectives(items):
    """Render a checklist from (label, is_done) pairs."""
    out = "".join(
        f"<div class='obj {'ok' if ok else ''}'>{'✅' if ok else '⬜'}&nbsp; {label}</div>" for label, ok in items
    )
    html(f"<div class='chapter' style='margin-top:6px'>Objectives</div>{out}")


def tile(value, label, kind=""):
    return f"<div class='tile {kind}'><b>{value}</b><span>{label}</span></div>"


def style_fig(fig, title="", height=400):
    """Apply the shared dark/transparent Plotly look to every chart."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter"),
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def hint_box(stage: int, text: str):
    """Show a hint button (costs XP) or, once requested, the hint itself."""
    if stage in st.session_state.hints:
        st.info(f"💡 **Hint:** {text}")
    elif stage not in st.session_state.done:
        if st.button(f"💡 Ask NOVA for a hint (-{HINT_PENALTY} XP reward)", key=f"hint{stage}"):
            st.session_state.hints.add(stage)
            st.rerun()


def mission_footer(stage: int, won: bool, success_msg: str):
    """Bottom of each mission: claim-reward button when `won`, or navigation once already completed."""
    if stage in st.session_state.done:
        html("<div class='victory'><h3>✅ SYSTEM REPAIRED</h3>Reward already banked.</div>")
        if stage < 4:
            st.button("Next mission ➜", key=f"next{stage}", on_click=go_to, args=(stage + 1,))
        else:
            st.button("Open Mission Log 🏆", key=f"next{stage}", on_click=go_to, args=(5,))
        return
    if won:
        reward = MISSION_REWARD - (HINT_PENALTY if stage in st.session_state.hints else 0)
        html(f"<div class='victory'><h3>🎉 OBJECTIVES COMPLETE</h3>{success_msg}</div>")
        st.button(f"⭐ Claim {reward} XP & repair system", key=f"claim{stage}", on_click=claim, args=(stage,))
    else:
        st.caption("Complete every objective to repair this system.")


def manual(title: str, body: str):
    with st.expander(f"📘 Field Manual: {title}"):
        st.markdown(body)


# ==========================================================
# DATA / MODEL CACHES
# ==========================================================
@st.cache_data
def signal_data():
    """Mission 1: a sine wave + noise with 3 NaNs and 3 planted outliers (indices 12, 35, 50)."""
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 60)
    truth = np.sin(x)
    y = truth + rng.normal(0, 0.15, 60)
    for i in (5, 20, 44):
        y[i] = np.nan
    y[12], y[35], y[50] = -4.2, 4.5, 3.8
    return x, truth, y


@st.cache_data
def threat_data():
    """Mission 2: train logistic regression on UCI Breast Cancer; label 1 = malignant ("threat").

    Returns the true labels and predicted threat probabilities. The model is fit on all
    30 features, so precision and recall can both be high at a sensible threshold.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = 1 - data.target  # 1 = threat (malignant), 0 = harmless
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(Xs, y)
    probs = model.predict_proba(Xs)[:, 1]
    return np.asarray(y), probs


@st.cache_data
def orbit_data():
    """Mission 3: noisy sine data randomly split into 24 training and 16 test points."""
    rng = np.random.default_rng(7)
    x = np.linspace(-3, 3, 40)
    y = np.sin(x) + rng.normal(0, 0.2, 40)
    idx = rng.permutation(40)
    return x[idx[:24]], y[idx[:24]], x[idx[24:]], y[idx[24:]]


def fit_orbit(deg, kind, alpha):
    """Fit a polynomial (+ optional Ridge/Lasso) model; return (pipeline, train RMSE, test RMSE)."""
    xtr, ytr, xte, yte = orbit_data()
    if alpha == 0 or kind == "None":
        reg = LinearRegression()
    elif kind == "Ridge (L2)":
        reg = Ridge(alpha=alpha)
    else:
        reg = Lasso(alpha=alpha, max_iter=100000)
    pipe = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(), reg)
    pipe.fit(xtr.reshape(-1, 1), ytr)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    tr = rmse(pipe.predict(xtr.reshape(-1, 1)), ytr)
    te = rmse(pipe.predict(xte.reshape(-1, 1)), yte)
    return pipe, tr, te


@st.cache_data
def orbit_target():
    """Winning test-error target: 15% worse than the best model over a grid of settings."""
    best = min(
        fit_orbit(d, k, a)[2]
        for d in range(1, 11)
        for k, a in (("None", 0), ("Ridge (L2)", 0.05), ("Ridge (L2)", 0.3), ("Lasso (L1)", 0.01))
    )
    return best * 1.15 + 0.01


@st.cache_data
def fleet_data():
    """Mission 4: 4 hidden blobs; precompute inertia and silhouette for K=2..8 and the best K."""
    pts, _ = make_blobs(n_samples=200, centers=[(-3, -2), (3, -2), (0, 3.2), (4.5, 4)], cluster_std=0.65, random_state=3)
    ks = list(range(2, 9))
    inertia, sil = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pts)
        inertia[k], sil[k] = km.inertia_, silhouette_score(pts, km.labels_)
    best_k = max(sil, key=sil.get)
    return pts, inertia, sil, best_k


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    html("<div class='logo'>STARFALL</div><div class='logo-sub'>A Data Science Adventure</div>")
    rank, nxt, pct = rank_info()
    html(
        f"<div class='hud-item'><div class='hud-label'>Rank</div><div class='hud-value'>{rank}</div>"
        f"<div class='bar'><div style='width:{pct}%'></div></div>"
        f"<div class='hud-label' style='margin-top:6px'>{st.session_state.xp} / {nxt} XP</div></div>"
    )
    st.write("")

    def stage_label(s):
        icon, name = STAGES[s]
        if s in st.session_state.done:
            return f"✅ {name}"
        if not is_unlocked(s):
            return f"🔒 {name}"
        return f"{icon} {name}"

    st.radio("Navigation", list(STAGES), key="nav", format_func=stage_label, label_visibility="collapsed")
    st.divider()
    st.button("↺ Restart game", on_click=reset_game)

stage = st.session_state.nav
hud()

if not is_unlocked(stage):
    html(
        "<div class='glass locked'><div class='big'>🔒</div><h2>Sector locked</h2>"
        "<p>Repair the previous system to unlock this mission.</p></div>"
    )
    st.stop()

# ==========================================================
# PROLOGUE
# ==========================================================
if stage == 0:
    html(
        "<div class='hero'><div class='chapter'>Stardate 2187 · Emergency</div>"
        "<h1 class='title-grad'>The Meridian is falling.</h1>"
        "<p style='font-size:1.05rem;color:#cbd5e1'>A solar storm has scrambled every system aboard your starship. "
        "Sensors lie, scanners miss threats, the navigation curve is broken and the escort fleet has vanished from the radar. "
        "You are the only Data Officer left awake.</p></div>"
    )
    nova(
        "Officer, you're awake! I'm <b>NOVA</b>, the ship's AI. My own models are damaged, "
        "so I need <i>your</i> judgement. Four systems must be repaired, and each one needs a different "
        "data-science skill. I'll guide you, but <b>you</b> make the calls."
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, name, skill) in zip(
        (c1, c2, c3, c4),
        (
            ("🧹", "Signal Scrub", "Data cleaning"),
            ("🎯", "Threat Scanner", "Classification"),
            ("🌀", "Orbit Tuner", "Regression"),
            ("🛸", "Fleet Finder", "Clustering"),
        ),
    ):
        col.markdown(
            f"<div class='glass' style='text-align:center'><div style='font-size:2rem'>{icon}</div>"
            f"<b>{name}</b><br><span style='color:#94a3b8'>{skill}</span></div>",
            unsafe_allow_html=True,
        )
    st.button("🚀 Begin Mission 1", on_click=go_to, args=(1,))

# ==========================================================
# MISSION 1: SIGNAL SCRUB
# ==========================================================
elif stage == 1:
    mission_header("Mission 1 · Data Cleaning", "Signal Scrub")
    nova(
        "The storm corrupted our life-support telemetry. Some readings are <b>missing (NaN)</b> and a few "
        "are wild <b>outliers</b> that would make the engine panic. Clean the feed so only the true "
        "wave remains, but don't throw away good data!"
    )
    x, truth, y_raw = signal_data()
    corrupted = {12, 35, 50}
    nan_idx = set(np.where(np.isnan(y_raw))[0])

    left, right = st.columns([1, 2])
    with left:
        st.subheader("🎛️ Controls")
        nan_strat = st.selectbox("1 · Missing readings", ["Leave them", "Drop those rows", "Fill with average"])
        z_cut = st.slider("2 · Outlier cutoff (σ from average)", 1.0, 4.0, 4.0, 0.1,
                          help="Readings further than this many standard deviations are removed. 4.0 = filter off.")
        scale = st.selectbox("3 · Rescale signal (optional)", ["None", "Min-Max (0 to 1)", "Standardize (mean 0, σ 1)"])

    df = pd.DataFrame({"x": x, "y": y_raw})
    if nan_strat == "Drop those rows":
        df = df.dropna()
    elif nan_strat == "Fill with average":
        df["y"] = df["y"].fillna(df["y"].mean())

    if z_cut < 4.0:
        z = (df["y"] - df["y"].mean()) / df["y"].std()
        df = df[(z.abs() <= z_cut) | z.isna()]

    nan_left = int(df["y"].isna().sum())
    outliers_left = len(corrupted & set(df.index))
    legit = set(range(len(x))) - corrupted - nan_idx
    good_lost = len(legit - set(df.index))

    shown = df.copy()
    if scale != "None" and not shown["y"].isna().any() and len(shown) > 1:
        scaler = MinMaxScaler() if scale.startswith("Min") else StandardScaler()
        shown["y"] = scaler.fit_transform(shown[["y"]])

    with right:
        fig = go.Figure()
        if scale == "None":
            fig.add_trace(go.Scatter(x=x, y=y_raw, mode="markers", name="Raw feed",
                                     marker=dict(color="rgba(148,163,184,0.35)", size=8)))
            fig.add_trace(go.Scatter(x=x, y=truth, mode="lines", name="True signal (hidden from the ship)",
                                     line=dict(color="#a78bfa", dash="dot")))
        fig.add_trace(go.Scatter(x=shown["x"], y=shown["y"], mode="markers", name="Your cleaned feed",
                                 marker=dict(color="#22d3ee", size=10, line=dict(color="white", width=1))))
        style_fig(fig, "Telemetry Feed", 400)
        st.plotly_chart(fig, use_container_width=True)

    c = st.columns(4)
    c[0].metric("Rows kept", f"{len(df)}/60")
    c[1].metric("Missing left", nan_left)
    c[2].metric("Outliers left", outliers_left)
    c[3].metric("Good data lost", good_lost)

    ok1 = nan_left == 0 and nan_strat != "Leave them"
    ok2 = outliers_left == 0
    ok3 = good_lost <= 3
    objectives([
        ("No missing readings remain", ok1),
        ("All 3 wild outliers removed", ok2),
        ("Lose at most 3 good readings", ok3),
    ])
    won = ok1 and ok2 and ok3
    hint_box(1, "Handle the missing values first. Then lower the cutoff slowly: too low (around 1σ) "
                "starts deleting the real wave peaks. Around 2 to 3σ is the sweet spot.")
    mission_footer(1, won, "The feed is clean and the real wave is intact. Life support is stable!")
    manual(
        "Cleaning data",
        "- **Missing values (NaN):** drop the rows (simple, but you lose data) or *impute* them (e.g. with the mean).\n"
        "- **Outliers:** a **z-score** says how many standard deviations a point is from the average. "
        "Points with a huge |z| are suspicious.\n"
        "- **Trade-off:** cleaning too hard removes real signal; too little leaves noise.\n"
        "- **Scaling** (Min-Max, Standardize) changes the *units* of the data, not its shape. It matters for "
        "distance-based and regularized models.",
    )

# ==========================================================
# MISSION 2: THREAT SCANNER
# ==========================================================
elif stage == 2:
    mission_header("Mission 2 · Classification", "Threat Scanner")
    nova(
        "Our bio-scanner assigns every sample a <b>threat probability</b>. You choose the "
        "<b>alarm threshold</b>: above it we sound the alarm. Too low and we cry wolf constantly. "
        "Too high and a real threat slips through. Find the balance! "
        "<span style='color:#94a3b8'>(Powered by the real UCI Breast Cancer dataset, reframed as ship scans.)</span>"
    )
    y_true, probs = threat_data()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("🎛️ Controls")
        if "tau" not in st.session_state:
            st.session_state.tau = 0.50
        st.caption("Quick presets")
        pc = st.columns(3)
        pc[0].button("😱 Paranoid", key="p_par", on_click=set_tau, args=(0.10,), help="Alarm on almost anything")
        pc[1].button("⚖️ Balanced", key="p_bal", on_click=set_tau, args=(0.50,), help="The default middle ground")
        pc[2].button("😴 Relaxed", key="p_rel", on_click=set_tau, args=(0.90,), help="Only alarm when very sure")
        tau = st.slider("Alarm threshold (τ)", 0.05, 0.95, step=0.05, key="tau")
        pred = (probs >= tau).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        html(
            "<div class='tiles'>"
            + tile(tp, "Threats caught ✔", "good")
            + tile(fp, "False alarms 🚨", "warn")
            + tile(fn, "Threats MISSED ☠", "bad")
            + tile(tn, "Correct all-clear ✔", "good")
            + "</div>"
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Precision", f"{prec:.0%}", help="Of all alarms, how many were real?")
        m2.metric("Recall", f"{rec:.0%}", help="Of all real threats, how many did we catch?")
        gauge = go.Figure()
        for name, val, dom in ("Precision", prec, [0, 0.48]), ("Recall", rec, [0.52, 1]):
            gauge.add_trace(go.Indicator(
                mode="gauge+number", value=val * 100, number={"suffix": "%", "font": {"size": 22}},
                title={"text": name, "font": {"size": 13}}, domain={"x": dom, "y": [0, 1]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22d3ee"},
                       "threshold": {"line": {"color": "#fbbf24", "width": 3}, "value": 97},
                       "bgcolor": "rgba(148,163,184,0.12)"}))
        gauge.update_layout(height=170, margin={"l": 10, "r": 10, "t": 30, "b": 0},
                            paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark")
        st.plotly_chart(gauge, use_container_width=True)
        st.caption("Yellow marks are roughly the targets (recall 97%, precision 96%).")

    with right:
        tab1, tab5, tab3, tab2 = st.tabs(["📊 Scanner readings", "🧬 Scan grid", "🧮 Confusion matrix", "📈 ROC curve"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=probs[y_true == 0], name="Harmless", marker_color="#34d399",
                                       opacity=0.75, xbins=dict(start=0, end=1, size=0.04)))
            fig.add_trace(go.Histogram(x=probs[y_true == 1], name="Real threat", marker_color="#f87171",
                                       opacity=0.75, xbins=dict(start=0, end=1, size=0.04)))
            fig.add_vline(x=tau, line_color="#fbbf24", line_width=3, annotation_text="alarm line")
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title="Predicted threat probability")
            fig.update_yaxes(title="Samples")
            style_fig(fig, "Everything right of the yellow line triggers the alarm", 400)
            st.plotly_chart(fig, use_container_width=True)
        with tab5:
            rng = np.random.default_rng(5)
            sel = rng.choice(len(probs), 150, replace=False)
            cells = []
            for i in sel:
                alarm, real = probs[i] >= tau, y_true[i] == 1
                if alarm and real:
                    col, sym, tip = "#34d399", "🎯", "Threat caught"
                elif alarm:
                    col, sym, tip = "#fbbf24", "🚨", "False alarm"
                elif real:
                    col, sym, tip = "#f87171", "☠", "THREAT MISSED"
                else:
                    col, sym, tip = "#334155", "·", "All clear"
                cells.append(
                    f"<div title='{tip}' style='width:34px;height:34px;display:grid;place-items:center;"
                    f"border-radius:8px;background:{col}33;border:1px solid {col};color:{col}'>{sym}</div>"
                )
            html("<div class='glass'><div style='display:flex;flex-wrap:wrap;gap:6px'>" + "".join(cells) + "</div></div>")
            st.caption("🎯 caught · 🚨 false alarm · ☠ missed threat · · all clear. 150 random scans, updating live.")
        with tab3:
            z = [[tn, fp], [fn, tp]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["No alarm", "ALARM"], y=["Harmless", "Real threat"], colorscale="Purples",
                text=[[str(v) for v in r] for r in z], texttemplate="%{text}", textfont={"size": 26}, showscale=False))
            fig.update_yaxes(autorange="reversed")
            style_fig(fig, "Confusion matrix: what happened vs what the scanner said", 400)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fpr, tpr, _ = roc_curve(y_true, probs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Scanner", line=dict(color="#22d3ee", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[rec], mode="markers", name="Your threshold",
                                     marker=dict(size=14, color="#fbbf24", line=dict(color="white", width=2))))
            fig.update_xaxes(title="False alarm rate")
            fig.update_yaxes(title="Threats caught (recall)")
            style_fig(fig, "ROC curve: the closer to the top-left, the better", 400)
            st.plotly_chart(fig, use_container_width=True)

    objectives([
        ("Recall ≥ 97%: catch almost every real threat", rec >= 0.97),
        ("Precision ≥ 96%: keep false alarms low", prec >= 0.96),
    ])
    hint_box(2, "Lowering τ catches more threats (recall up) but causes more false alarms (precision down). "
                "The default 0.50 is too strict on recall. Try 0.30 to 0.50.")
    mission_footer(2, rec >= 0.97 and prec >= 0.96, "Threats caught, crew calm. The scanner is calibrated!")
    manual(
        "Classification metrics",
        "- **Precision** = real threats ÷ all alarms. *\"When it screams, is it right?\"*\n"
        "- **Recall** = real threats caught ÷ all real threats. *\"Does it miss anything?\"*\n"
        "- **Threshold:** a classifier outputs a probability; *you* pick the cut-off. Moving it trades "
        "precision for recall.\n"
        "- **ROC curve:** shows that trade-off at *every* threshold. More area under it (AUC) = better model.\n"
        "- In medicine, **missing a threat is usually worse** than a false alarm, so recall is prioritised.",
    )

# ==========================================================
# MISSION 3: ORBIT TUNER
# ==========================================================
elif stage == 3:
    mission_header("Mission 3 · Regression", "Orbit Tuner")
    nova(
        "Navigation must predict the ship's orbit from noisy readings. I fit a <b>polynomial curve</b> through "
        "the <span style='color:#22d3ee'>training points</span>, but what matters is how well it predicts "
        "the <span style='color:#fbbf24'>unseen test points</span>. Too simple <i>underfits</i>. "
        "Too wiggly <i>overfits</i>. Tune it!"
    )
    xtr, ytr, xte, yte = orbit_data()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("🎛️ Controls")
        deg = st.slider("Curve complexity (polynomial degree)", 1, 10, 1)
        kind = st.radio("Brake on complexity (regularization)", ["None", "Ridge (L2)", "Lasso (L1)"])
        alpha = st.slider("Brake strength (λ)", 0.0, 1.0, 0.0, 0.01, disabled=(kind == "None"))
        if kind == "None":
            alpha = 0.0

    pipe, tr_err, te_err = fit_orbit(deg, kind, alpha)
    target = orbit_target()
    if tr_err > 0.4:
        verdict, vcolor = "UNDERFITTING: the curve is too simple", "#fbbf24"
    elif te_err > tr_err * 1.6 and te_err > target:
        verdict, vcolor = "OVERFITTING: memorising noise", "#f87171"
    elif te_err <= target:
        verdict, vcolor = "SWEET SPOT: generalises well", "#34d399"
    else:
        verdict, vcolor = "Getting close... keep tuning", "#94a3b8"

    with right:
        xs = np.linspace(-3, 3, 300)
        ys = pipe.predict(xs.reshape(-1, 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=np.sin(xs), mode="lines", name="True orbit (unknown)",
                                 line=dict(color="rgba(167,139,250,0.7)", dash="dot")))
        fig.add_trace(go.Scatter(x=xs, y=np.clip(ys, -3, 3), mode="lines", name="Your model",
                                 line=dict(color="#22d3ee", width=4)))
        fig.add_trace(go.Scatter(x=xtr, y=ytr, mode="markers", name="Training data",
                                 marker=dict(color="#22d3ee", size=9)))
        fig.add_trace(go.Scatter(x=xte, y=yte, mode="markers", name="Test data (unseen)",
                                 marker=dict(color="#fbbf24", size=10, symbol="diamond")))
        fig.update_yaxes(range=[-2.5, 2.5])
        style_fig(fig, "Orbit prediction", 400)
        st.plotly_chart(fig, use_container_width=True)

    html(f"<div class='glass' style='border-color:{vcolor};text-align:center;font-family:Orbitron'>{verdict}</div>")
    m = st.columns(3)
    m[0].metric("Training error", f"{tr_err:.3f}")
    m[1].metric("Test error", f"{te_err:.3f}", delta=f"{te_err - tr_err:+.3f} vs train", delta_color="inverse")
    m[2].metric("Target test error", f"≤ {target:.3f}")

    objectives([(f"Test error ≤ {target:.3f}", te_err <= target)])
    hint_box(3, "Degree 1 is a straight line, far too simple. Raise the degree until the curve follows the wave, "
                "and if it starts to wiggle wildly on the yellow diamonds, add a bit of Ridge regularization.")
    mission_footer(3, te_err <= target, "The orbit model predicts unseen data. Navigation is back online!")
    manual(
        "Under- vs overfitting",
        "- **Underfitting:** model too simple → high error on *both* training and test data.\n"
        "- **Overfitting:** model too flexible → tiny training error but **large test error**; it memorised noise.\n"
        "- **Regularization** adds a penalty on large coefficients. **Ridge (L2)** shrinks them smoothly; "
        "**Lasso (L1)** can push some to exactly zero.\n"
        "- Always judge a model on data it has **never seen**.",
    )

# ==========================================================
# MISSION 4: FLEET FINDER
# ==========================================================
elif stage == 4:
    mission_header("Mission 4 · Clustering", "Fleet Finder")
    nova(
        "Our allied fleets are out there, but radar only shows <b>unlabelled blips</b>. Nobody told me how many "
        "fleets exist! Use <b>K-Means</b> to group the blips. Your task: discover the <b>right number of "
        "clusters (K)</b> using the intel charts."
    )
    pts, inertia, sil, best_k = fleet_data()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("🎛️ Controls")
        k = st.slider("Number of fleets (K)", 1, 8, 1)
        if k > 1:
            st.metric("Cluster quality (silhouette)", f"{sil[k]:.2f}", help="Closer to 1 = tight, well separated groups")
        else:
            st.metric("Cluster quality (silhouette)", "n/a", help="Needs at least 2 clusters")
        st.caption("Tip: open the intel tabs to compare different K values.")

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(pts)
    palette = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#a78bfa", "#fb923c", "#34d399", "#f87171"]

    with right:
        tab1, tab2, tab3 = st.tabs(["📡 Radar", "📉 Elbow chart", "📊 Silhouette chart"])
        with tab1:
            fig = go.Figure()
            for c in range(k):
                m = labels == c
                fig.add_trace(go.Scatter(x=pts[m, 0], y=pts[m, 1], mode="markers", name=f"Fleet {c + 1}",
                                         marker=dict(color=palette[c], size=9, line=dict(color="rgba(255,255,255,.5)", width=1))))
            style_fig(fig, f"Radar view · K = {k}", 400)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = go.Figure(go.Scatter(x=list(inertia), y=list(inertia.values()), mode="lines+markers",
                                       line=dict(color="#22d3ee", width=3), marker=dict(size=10)))
            fig.update_xaxes(title="K")
            fig.update_yaxes(title="Inertia (spread inside clusters)")
            style_fig(fig, "Look for the 'elbow' where improvement flattens", 400)
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            fig = go.Figure(go.Bar(x=list(sil), y=list(sil.values()),
                                   marker_color=["#34d399" if kk == best_k else "#475569" for kk in sil]))
            fig.update_xaxes(title="K")
            fig.update_yaxes(title="Silhouette score")
            style_fig(fig, "Higher is better: the tallest bar is the best K", 400)
            st.plotly_chart(fig, use_container_width=True)

    objectives([("Choose the K with the best silhouette score", k == best_k)])
    hint_box(4, "Check the Elbow chart for the bend, and the Silhouette chart for the tallest bar. "
                "Both point at the same K.")
    mission_footer(4, k == best_k, f"Exactly {best_k} fleets detected. Escort coordinates locked in!")
    manual(
        "Clustering",
        "- **Unsupervised learning:** no labels, so the algorithm finds structure on its own.\n"
        "- **K-Means** places K centres and assigns each point to the nearest one, then repeats.\n"
        "- **Elbow method:** plot within-cluster spread (inertia) against K; pick the bend.\n"
        "- **Silhouette score:** −1 to 1. How close each point is to its own cluster compared with others. "
        "Higher means better-separated clusters.",
    )

# ==========================================================
# DEBRIEF
# ==========================================================
else:
    html(
        "<div class='hero'><div class='chapter'>All systems online</div>"
        "<h1 class='title-grad'>The Meridian is saved!</h1>"
        "<p style='font-size:1.05rem;color:#cbd5e1'>Thanks to your data skills the ship is back on course. "
        "Here is what you mastered:</p></div>"
    )
    rank, _, _ = rank_info()
    nova(f"Outstanding work, <b>{rank}</b>! You earned <b>{st.session_state.xp} XP</b>. "
         "Everything you did today is what real data scientists do every day.")
    rows = [
        ("🧹 Signal Scrub", "Data cleaning", "Handle NaNs, filter outliers with z-scores, scale features"),
        ("🎯 Threat Scanner", "Classification", "Thresholds, precision vs recall, ROC curve"),
        ("🌀 Orbit Tuner", "Regression", "Under/overfitting, Ridge & Lasso regularization"),
        ("🛸 Fleet Finder", "Clustering", "K-Means, elbow method, silhouette score"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Mission", "Topic", "You learned"]), hide_index=True, use_container_width=True)
    if st.session_state.xp == MISSION_REWARD * 4:
        st.success("🏅 Perfect run: no hints used!")
    st.button("↺ Play again", on_click=reset_game, key="again")
