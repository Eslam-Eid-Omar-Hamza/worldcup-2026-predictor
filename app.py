import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==================== CONFIG ====================
st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CUTOFF = "2018-01-01"

# ==================== TRANSLATIONS ====================
T = {
    "en": {
        "title": "⚽ Match Predictor",
        "subtitle": "Poisson + Elo model · live data · international football",
        "lang_label": "🌐 العربية",
        "home": "Team A",
        "away": "Team B",
        "knockout": "Knockout match (extra time & penalties)",
        "predict": "Predict",
        "loading": "Pulling latest data & training model...",
        "pick": "Pick two different teams.",
        "win": "win",
        "draw": "Draw",
        "exp_goals": "Expected goals",
        "in90": "In 90 minutes",
        "decided90": "Decided in 90 min",
        "extratime": "Goes to extra time",
        "penalties": "Goes to penalties",
        "qualify": "Who qualifies?",
        "strengths": "Team strengths",
        "attack": "Attack",
        "defense": "Defense",
        "elo": "Elo rating",
        "vs_avg": "vs average",
        "strong": "strong", "avg": "average", "weak": "weak",
        "updated": "Data updated to",
        "disclaimer": "Football is unpredictable — these are probabilities from historical form, not certainties.",
    },
    "ar": {
        "title": "⚽ متنبّئ نتايج الماتشات",
        "subtitle": "نموذج Poisson + Elo · داتا حيّة · كرة قدم دولية",
        "lang_label": "🌐 English",
        "home": "الفريق الأول",
        "away": "الفريق التاني",
        "knockout": "ماتش خروج مغلوب (وقت إضافي وجزا)",
        "predict": "احسب التوقّع",
        "loading": "بنسحب آخر داتا وبندرّب الموديل...",
        "pick": "اختار فريقين مختلفين.",
        "win": "يكسب",
        "draw": "تعادل",
        "exp_goals": "الأهداف المتوقّعة",
        "in90": "في الـ 90 دقيقة",
        "decided90": "يتحسم في 90 دقيقة",
        "extratime": "يروح وقت إضافي",
        "penalties": "يوصل ضربات جزا",
        "qualify": "مين يتأهّل؟",
        "strengths": "قوة الفريقين",
        "attack": "الهجوم",
        "defense": "الدفاع",
        "elo": "تصنيف Elo",
        "vs_avg": "مقارنة بالمتوسط",
        "strong": "قوي", "avg": "متوسط", "weak": "ضعيف",
        "updated": "الداتا محدّثة حتى",
        "disclaimer": "الكورة مبتتحسبش — دي احتمالات من الأداء التاريخي، مش نتايج مؤكدة.",
    },
}

# ==================== MODEL (cached 1 hour) ====================
@st.cache_data(ttl=3600)
def load_and_train():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    recent = df[df["date"] >= CUTOFF].copy()
    latest_date = df["date"].max()

    overall_avg = (recent["home_score"].mean() + recent["away_score"].mean()) / 2

    teams = pd.unique(recent[["home_team", "away_team"]].values.ravel())
    rows = []
    for t in teams:
        h = recent[recent["home_team"] == t]
        a = recent[recent["away_team"] == t]
        g = len(h) + len(a)
        if g < 5:
            continue
        gf = h["home_score"].sum() + a["away_score"].sum()
        ga = h["away_score"].sum() + a["home_score"].sum()
        rows.append({"team": t, "attack": (gf/g)/overall_avg, "defense": (ga/g)/overall_avg})
    S = pd.DataFrame(rows).set_index("team")

    # Elo
    elo = {t: 1500.0 for t in teams}
    K = 30
    for _, r in recent.sort_values("date").iterrows():
        h, a = r["home_team"], r["away_team"]
        Rh, Ra = elo[h], elo[a]
        neutral = (r["neutral"] == True) or (str(r["neutral"]).upper() == "TRUE")
        ha = 0 if neutral else 60
        Eh = 1 / (1 + 10 ** ((Ra - (Rh + ha)) / 400))
        if r["home_score"] > r["away_score"]: Sh = 1
        elif r["home_score"] < r["away_score"]: Sh = 0
        else: Sh = 0.5
        gd = abs(r["home_score"] - r["away_score"])
        mult = 1 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 2))
        elo[h] = Rh + K*mult*(Sh - Eh)
        elo[a] = Ra + K*mult*((1-Sh) - (1-Eh))
    for t in S.index:
        S.loc[t, "elo"] = round(elo[t])

    return S, overall_avg, latest_date

def predict(S, overall_avg, home, away, max_goals=10):
    la_h = overall_avg * S.loc[home,"attack"] * S.loc[away,"defense"]
    la_a = overall_avg * S.loc[away,"attack"] * S.loc[home,"defense"]
    m = np.outer(poisson.pmf(range(max_goals), la_h), poisson.pmf(range(max_goals), la_a))
    p_h = np.tril(m,-1).sum(); p_d = np.trace(m); p_a = np.triu(m,1).sum()
    # extra time
    m_et = np.outer(poisson.pmf(range(max_goals), la_h/3), poisson.pmf(range(max_goals), la_a/3))
    p_h_et = np.tril(m_et,-1).sum(); p_d_et = np.trace(m_et); p_a_et = np.triu(m_et,1).sum()
    goes_et = p_d; goes_pens = p_d*p_d_et
    final_h = p_h + p_d*(p_h_et + p_d_et*0.5)
    final_a = p_a + p_d*(p_a_et + p_d_et*0.5)
    return la_h, la_a, p_h, p_d, p_a, goes_et, goes_pens, final_h, final_a

def strength_label(tr, val, is_defense=False):
    # for defense lower is better -> invert
    v = (1/val) if is_defense else val
    if v >= 1.15: return tr["strong"]
    if v <= 0.85: return tr["weak"]
    return tr["avg"]

# ==================== UI ====================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

tr = T[st.session_state.lang]
rtl = st.session_state.lang == "ar"

# language toggle
c1, c2 = st.columns([4,1])
with c2:
    if st.button(tr["lang_label"]):
        st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
        st.rerun()

tr = T[st.session_state.lang]
rtl = st.session_state.lang == "ar"

if rtl:
    st.markdown('<style>.main * { direction: rtl; text-align: right; }</style>', unsafe_allow_html=True)

st.title(tr["title"])
st.caption(tr["subtitle"])

with st.spinner(tr["loading"]):
    S, overall_avg, latest_date = load_and_train()

teams_sorted = sorted(S.index.tolist())

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox(tr["home"], teams_sorted,
                        index=teams_sorted.index("Egypt") if "Egypt" in teams_sorted else 0)
with col2:
    away = st.selectbox(tr["away"], teams_sorted,
                        index=teams_sorted.index("Argentina") if "Argentina" in teams_sorted else 1)

knockout = st.checkbox(tr["knockout"], value=True)

if st.button(tr["predict"], type="primary", use_container_width=True):
    if home == away:
        st.warning(tr["pick"])
    else:
        la_h, la_a, p_h, p_d, p_a, goes_et, goes_pens, final_h, final_a = predict(S, overall_avg, home, away)

        st.divider()
        # main probabilities
        st.subheader(tr["in90"])
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} {tr['win']}", f"{p_h*100:.0f}%")
        m2.metric(tr["draw"], f"{p_d*100:.0f}%")
        m3.metric(f"{away} {tr['win']}", f"{p_a*100:.0f}%")

        g1, g2 = st.columns(2)
        g1.metric(f"{tr['exp_goals']} · {home}", f"{la_h:.2f}")
        g2.metric(f"{tr['exp_goals']} · {away}", f"{la_a:.2f}")

        if knockout:
            st.subheader("⏱️ " + tr["qualify"])
            k1, k2, k3 = st.columns(3)
            k1.metric(tr["decided90"], f"{(p_h+p_a)*100:.0f}%")
            k2.metric(tr["extratime"], f"{goes_et*100:.0f}%")
            k3.metric(tr["penalties"], f"{goes_pens*100:.0f}%")
            q1, q2 = st.columns(2)
            q1.metric(f"{home}", f"{final_h*100:.0f}%")
            q2.metric(f"{away}", f"{final_a*100:.0f}%")

        # strengths
        st.subheader("💪 " + tr["strengths"])
        for team in [home, away]:
            atk = S.loc[team,"attack"]; dfn = S.loc[team,"defense"]; elo = int(S.loc[team,"elo"])
            atk_lbl = strength_label(tr, atk)
            dfn_lbl = strength_label(tr, dfn, is_defense=True)
            st.markdown(
                f"**{team}** — {tr['attack']}: {atk:.2f} ({atk_lbl}) · "
                f"{tr['defense']}: {dfn:.2f} ({dfn_lbl}) · {tr['elo']}: {elo}"
            )

        st.divider()
        st.caption(f"📅 {tr['updated']} {latest_date.date()}")
        st.caption(f"ℹ️ {tr['disclaimer']}")
