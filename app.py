import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import country_converter as coco
import logging
logging.getLogger("country_converter").setLevel(logging.CRITICAL)

# ==================== CONFIG ====================
st.set_page_config(page_title="Match Predictor", page_icon="⚽", layout="centered")
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CUTOFF = "2018-01-01"

# ==================== ARABIC NAME MAP ====================
# Maps Arabic team names -> English (as they appear in the data)
AR_NAMES = {
    "مصر":"Egypt","الأرجنتين":"Argentina","البرازيل":"Brazil","السعودية":"Saudi Arabia",
    "المغرب":"Morocco","الجزائر":"Algeria","تونس":"Tunisia","قطر":"Qatar",
    "الإمارات":"United Arab Emirates","العراق":"Iraq","الأردن":"Jordan","الكويت":"Kuwait",
    "عمان":"Oman","البحرين":"Bahrain","لبنان":"Lebanon","سوريا":"Syria","فلسطين":"Palestine",
    "ليبيا":"Libya","السودان":"Sudan","اليمن":"Yemen","فرنسا":"France","ألمانيا":"Germany",
    "إسبانيا":"Spain","البرتغال":"Portugal","إنجلترا":"England","إيطاليا":"Italy",
    "هولندا":"Netherlands","بلجيكا":"Belgium","كرواتيا":"Croatia","السنغال":"Senegal",
    "نيجيريا":"Nigeria","غانا":"Ghana","الكاميرون":"Cameroon","اليابان":"Japan",
    "كوريا الجنوبية":"South Korea","أستراليا":"Australia","أمريكا":"United States",
    "المكسيك":"Mexico","الأوروجواي":"Uruguay","كولومبيا":"Colombia","إيران":"Iran",
}
EN_TO_AR = {v:k for k,v in AR_NAMES.items()}

# ==================== FLAGS ====================
SPECIAL_FLAGS = {
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Wales":"🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Northern Ireland":"🇬🇧","Kosovo":"🇽🇰",
}
@st.cache_data
def get_flag(name):
    if name in SPECIAL_FLAGS: return SPECIAL_FLAGS[name]
    try:
        iso2 = coco.convert(names=name, to="ISO2")
        if iso2 and iso2 != "not found" and len(iso2) == 2:
            return chr(0x1F1E6+ord(iso2[0])-ord("A"))+chr(0x1F1E6+ord(iso2[1])-ord("A"))
    except Exception: pass
    return "🏳️"

# ==================== TRANSLATIONS ====================
T = {
 "en":{"title":"Match Predictor","subtitle":"Live international football predictions",
  "lang_label":"🌐 العربية","home":"Team A","away":"Team B",
  "knockout":"Knockout match (extra time & penalties)","predict":"⚽ Predict Match",
  "loading":"Pulling latest data & training model...","pick":"⚠️ Please pick two different teams.",
  "win":"win","draw":"Draw","exp_goals":"Expected goals (avg)",
  "in90":"📊 Win probability (90 minutes)","decided90":"Decided in 90'",
  "extratime":"Extra time","penalties":"Penalties","knockout_head":"⏱️ If it's a knockout",
  "qualify":"🏆 Who goes through","strengths":"💪 Team strengths",
  "attack":"Attack","defense":"Defense","elo":"Rating",
  "strong":"strong","avg":"average","weak":"weak","updated":"Data updated to",
  "disclaimer":"Football is unpredictable — these are probabilities from past form, not certainties. Made for fun ⚽",
  "verdict_fav":"**{a}** are the favourites, but **{b}** can cause an upset.",
  "verdict_close":"This is a coin-flip — **{a}** and **{b}** are very evenly matched!",
  "pens_note":"High chance of penalties 🥅 — anything can happen in a shootout!"},
 "ar":{"title":"متنبّئ الماتشات","subtitle":"توقّعات حيّة لكرة القدم الدولية",
  "lang_label":"🌐 English","home":"الفريق الأول","away":"الفريق التاني",
  "knockout":"ماتش خروج مغلوب (وقت إضافي وجزا)","predict":"⚽ احسب التوقّع",
  "loading":"بنسحب آخر داتا وبندرّب الموديل...","pick":"⚠️ اختار فريقين مختلفين.",
  "win":"يكسب","draw":"تعادل","exp_goals":"متوسط الأهداف المتوقّع",
  "in90":"📊 احتمالات الفوز (في الـ 90 دقيقة)","decided90":"يتحسم في 90'",
  "extratime":"وقت إضافي","penalties":"ضربات جزا","knockout_head":"⏱️ لو خروج مغلوب",
  "qualify":"🏆 مين يتأهّل","strengths":"💪 قوة الفريقين",
  "attack":"الهجوم","defense":"الدفاع","elo":"التصنيف",
  "strong":"قوي","avg":"متوسط","weak":"ضعيف","updated":"الداتا محدّثة حتى",
  "disclaimer":"الكورة مبتتحسبش — دي احتمالات من الأداء السابق مش نتايج مؤكدة. اتعملت للتسلية ⚽",
  "verdict_fav":"**{a}** هو المرشّح، بس **{b}** ممكن يعمل مفاجأة.",
  "verdict_close":"الماتش ونص — **{a}** و **{b}** متقاربين جداً!",
  "pens_note":"احتمال كبير لضربات الجزا 🥅 — وفيها أي حاجة ممكن تحصل!"},
}

# ==================== MODEL ====================
@st.cache_data(ttl=3600)
def load_and_train():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score","away_score"]).copy()
    df["home_score"]=df["home_score"].astype(int); df["away_score"]=df["away_score"].astype(int)
    recent = df[df["date"]>=CUTOFF].copy()
    latest_date = df["date"].max()
    overall_avg = (recent["home_score"].mean()+recent["away_score"].mean())/2
    teams = pd.unique(recent[["home_team","away_team"]].values.ravel())
    rows=[]
    for t in teams:
        h=recent[recent["home_team"]==t]; a=recent[recent["away_team"]==t]; g=len(h)+len(a)
        if g<5: continue
        gf=h["home_score"].sum()+a["away_score"].sum(); ga=h["away_score"].sum()+a["home_score"].sum()
        rows.append({"team":t,"attack":(gf/g)/overall_avg,"defense":(ga/g)/overall_avg})
    S=pd.DataFrame(rows).set_index("team")
    elo={t:1500.0 for t in teams}; K=30
    for _,r in recent.sort_values("date").iterrows():
        h,a=r["home_team"],r["away_team"]; Rh,Ra=elo[h],elo[a]
        neutral=(r["neutral"]==True) or (str(r["neutral"]).upper()=="TRUE")
        ha=0 if neutral else 60
        Eh=1/(1+10**((Ra-(Rh+ha))/400))
        if r["home_score"]>r["away_score"]: Sh=1
        elif r["home_score"]<r["away_score"]: Sh=0
        else: Sh=0.5
        gd=abs(r["home_score"]-r["away_score"])
        mult=1 if gd<=1 else (1.5 if gd==2 else (1.75 if gd==3 else 2))
        elo[h]=Rh+K*mult*(Sh-Eh); elo[a]=Ra+K*mult*((1-Sh)-(1-Eh))
    for t in S.index: S.loc[t,"elo"]=round(elo[t])
    return S, overall_avg, latest_date

def predict(S,avg,home,away,mg=10):
    lh=avg*S.loc[home,"attack"]*S.loc[away,"defense"]
    la=avg*S.loc[away,"attack"]*S.loc[home,"defense"]
    m=np.outer(poisson.pmf(range(mg),lh),poisson.pmf(range(mg),la))
    ph,pd_,pa=np.tril(m,-1).sum(),np.trace(m),np.triu(m,1).sum()
    met=np.outer(poisson.pmf(range(mg),lh/3),poisson.pmf(range(mg),la/3))
    phe,pde,pae=np.tril(met,-1).sum(),np.trace(met),np.triu(met,1).sum()
    return lh,la,ph,pd_,pa,pd_,pd_*pde,ph+pd_*(phe+pde*0.5),pa+pd_*(pae+pde*0.5)

def slabel(tr,val,d=False):
    v=(1/val) if d else val
    return tr["strong"] if v>=1.15 else (tr["weak"] if v<=0.85 else tr["avg"])

def display_name(team, lang):
    # show Arabic name if available and lang is arabic, else English
    if lang=="ar" and team in EN_TO_AR:
        return EN_TO_AR[team]
    return team

# ==================== STYLING ====================
st.markdown("""
<style>
.stApp{background:linear-gradient(160deg,#0d1b2a 0%,#1b263b 100%);}
h1,h2,h3,h4,p,label,span,div{color:#e8eef5 !important;}
.block-container{padding-top:1.5rem;max-width:760px;}
[data-testid="stMetricValue"]{font-size:2rem !important;font-weight:800 !important;}
.stButton>button{background:linear-gradient(90deg,#00b894,#00cec9);color:#003;font-weight:800;
  border:none;border-radius:12px;padding:.6rem 1rem;font-size:1.05rem;}
.stButton>button:hover{filter:brightness(1.1);}
.hero{text-align:center;padding:6px 0 2px;}
.hero .ball{font-size:2.6rem;}
.matchup{display:flex;align-items:center;justify-content:center;gap:14px;margin:14px 0;}
.team-card{background:rgba(255,255,255,.06);border-radius:14px;padding:12px 18px;text-align:center;min-width:120px;}
.team-flag{font-size:2.4rem;line-height:1;}
.team-name{font-weight:700;margin-top:4px;font-size:1rem;}
.vs{font-size:1.3rem;font-weight:800;opacity:.6;}
.verdict{background:rgba(255,255,255,.07);border-left:4px solid #00b894;
  padding:14px 18px;border-radius:10px;font-size:1.1rem;margin:10px 0;}
.bar-wrap{display:flex;height:38px;border-radius:10px;overflow:hidden;margin:8px 0 4px;}
.bar-seg{display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.95rem;color:#fff;}
</style>
""", unsafe_allow_html=True)

# ==================== UI ====================
if "lang" not in st.session_state: st.session_state.lang="en"
tr=T[st.session_state.lang]

c1,c2=st.columns([4,1])
with c2:
    if st.button(tr["lang_label"]):
        st.session_state.lang="ar" if st.session_state.lang=="en" else "en"; st.rerun()
tr=T[st.session_state.lang]; lang=st.session_state.lang
if lang=="ar":
    st.markdown('<style>.main *{direction:rtl;text-align:right;}</style>',unsafe_allow_html=True)

st.markdown(f"<div class='hero'><div class='ball'>⚽</div><h1 style='margin:0;'>{tr['title']}</h1>"
            f"<p style='opacity:.7;margin-top:2px;'>{tr['subtitle']}</p></div>",unsafe_allow_html=True)

with st.spinner(tr["loading"]):
    S,avg,latest=load_and_train()
teams=sorted(S.index.tolist())

# selectbox WITHOUT flags in the label (clean typing) — Arabic display when available
col1,col2=st.columns(2)
with col1:
    home=st.selectbox(tr["home"],teams,
        index=teams.index("Egypt") if "Egypt" in teams else 0,
        format_func=lambda t: display_name(t,lang))
with col2:
    away=st.selectbox(tr["away"],teams,
        index=teams.index("Argentina") if "Argentina" in teams else 1,
        format_func=lambda t: display_name(t,lang))
knockout=st.checkbox(tr["knockout"],value=True)

# live matchup preview with flags OUTSIDE (next to result area)
fh,fa=get_flag(home),get_flag(away)
st.markdown(
    f"<div class='matchup'>"
    f"<div class='team-card'><div class='team-flag'>{fh}</div><div class='team-name'>{display_name(home,lang)}</div></div>"
    f"<div class='vs'>VS</div>"
    f"<div class='team-card'><div class='team-flag'>{fa}</div><div class='team-name'>{display_name(away,lang)}</div></div>"
    f"</div>", unsafe_allow_html=True)

if st.button(tr["predict"],type="primary",use_container_width=True):
    if home==away:
        st.warning(tr["pick"])
    else:
        hn,an=display_name(home,lang),display_name(away,lang)
        lh,la,ph,pd_,pa,g_et,g_pen,fin_h,fin_a=predict(S,avg,home,away)
        diff=abs(ph-pa)
        if diff<0.08:
            verdict=tr["verdict_close"].format(a=f"{fh} {hn}",b=f"{fa} {an}")
        else:
            if ph>pa: verdict=tr["verdict_fav"].format(a=f"{fh} {hn}",b=f"{fa} {an}")
            else: verdict=tr["verdict_fav"].format(a=f"{fa} {an}",b=f"{fh} {hn}")
        st.markdown(f"<div class='verdict'>{verdict}</div>",unsafe_allow_html=True)

        st.markdown(f"#### {tr['in90']}")
        st.markdown(
            f"<div class='bar-wrap'>"
            f"<div class='bar-seg' style='width:{ph*100:.0f}%;background:#d63031;'>{ph*100:.0f}%</div>"
            f"<div class='bar-seg' style='width:{pd_*100:.0f}%;background:#636e72;'>{pd_*100:.0f}%</div>"
            f"<div class='bar-seg' style='width:{pa*100:.0f}%;background:#0984e3;'>{pa*100:.0f}%</div></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:.9rem;opacity:.85;'>"
            f"<span>{fh} {hn} {tr['win']}</span><span>{tr['draw']}</span><span>{an} {tr['win']} {fa}</span></div>",
            unsafe_allow_html=True)

        g1,g2=st.columns(2)
        g1.metric(f"{fh} {hn} · {tr['exp_goals']}",f"{lh:.2f}")
        g2.metric(f"{fa} {an} · {tr['exp_goals']}",f"{la:.2f}")

        if knockout:
            st.markdown(f"#### {tr['knockout_head']}")
            k1,k2,k3=st.columns(3)
            k1.metric(tr["decided90"],f"{(ph+pa)*100:.0f}%")
            k2.metric(tr["extratime"],f"{g_et*100:.0f}%")
            k3.metric(tr["penalties"],f"{g_pen*100:.0f}%")
            if g_pen>0.15: st.info(tr["pens_note"])
            st.markdown(f"#### {tr['qualify']}")
            q1,q2=st.columns(2)
            q1.metric(f"{fh} {hn}",f"{fin_h*100:.0f}%")
            q2.metric(f"{fa} {an}",f"{fin_a*100:.0f}%")

        st.markdown(f"#### {tr['strengths']}")
        for team,flg in [(home,fh),(away,fa)]:
            atk=S.loc[team,"attack"]; dfn=S.loc[team,"defense"]; el=int(S.loc[team,"elo"])
            st.markdown(
                f"<div style='background:rgba(255,255,255,.05);padding:10px 14px;border-radius:10px;margin:5px 0;'>"
                f"<b>{flg} {display_name(team,lang)}</b> &nbsp;·&nbsp; {tr['attack']}: <b>{atk:.2f}</b> ({slabel(tr,atk)}) "
                f"&nbsp;·&nbsp; {tr['defense']}: <b>{dfn:.2f}</b> ({slabel(tr,dfn,True)}) "
                f"&nbsp;·&nbsp; {tr['elo']}: <b>{el}</b></div>",unsafe_allow_html=True)

        st.divider()
        st.caption(f"📅 {tr['updated']} {latest.date()}  ·  {tr['disclaimer']}")
