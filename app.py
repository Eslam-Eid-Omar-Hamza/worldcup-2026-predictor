import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import country_converter as coco
import urllib.request, json
import io
from PIL import Image, ImageDraw, ImageFont
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _AR_OK=True
except Exception:
    _AR_OK=False
import logging
logging.getLogger("country_converter").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Match Predictor", page_icon="⚽", layout="centered")

# ==================== DATA SOURCES ====================
NAT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SEASONS = ["2021-22","2022-23","2023-24","2024-25","2025-26"]

@st.cache_data(ttl=3600)
def load_national():
    df = pd.read_csv(NAT_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score","away_score"]).copy()
    df["home_score"]=df["home_score"].astype(int); df["away_score"]=df["away_score"].astype(int)
    df = df[df["date"] >= "2018-01-01"].copy()
    return df[["date","home_team","away_team","home_score","away_score","neutral"]]

@st.cache_data(ttl=3600)
def load_openfootball(code):
    base = "https://raw.githubusercontent.com/openfootball/football.json/master/{s}/"+code+".json"
    rows=[]
    for s in SEASONS:
        try:
            req=urllib.request.Request(base.format(s=s),headers={'User-Agent':'Mozilla/5.0'})
            data=json.load(urllib.request.urlopen(req,timeout=20))
            for m in data['matches']:
                sc=m.get('score')
                if isinstance(sc,dict) and sc.get('ft') and len(sc['ft'])==2:
                    rows.append({'date':m['date'],'home_team':m['team1'],'away_team':m['team2'],
                                 'home_score':sc['ft'][0],'away_score':sc['ft'][1],'neutral':False})
        except Exception: pass
    df=pd.DataFrame(rows); df['date']=pd.to_datetime(df['date'])
    df['home_score']=df['home_score'].astype(int); df['away_score']=df['away_score'].astype(int)
    return df

def get_data(league):
    if league=="national": return load_national()
    if league=="laliga": return load_openfootball("es.1")
    if league=="epl": return load_openfootball("en.1")
    if league=="bundesliga": return load_openfootball("de.1")

# ==================== MODEL ====================
@st.cache_data(ttl=3600)
def train(league):
    df = get_data(league)
    latest = df["date"].max()
    avg = (df["home_score"].mean()+df["away_score"].mean())/2
    teams = pd.unique(df[["home_team","away_team"]].values.ravel())
    rows=[]
    for t in teams:
        h=df[df["home_team"]==t]; a=df[df["away_team"]==t]; g=len(h)+len(a)
        if g<5: continue
        gf=h["home_score"].sum()+a["away_score"].sum(); ga=h["away_score"].sum()+a["home_score"].sum()
        rows.append({"team":t,"attack":(gf/g)/avg,"defense":(ga/g)/avg})
    S=pd.DataFrame(rows).set_index("team")
    elo={t:1500.0 for t in teams}; K=30
    for _,r in df.sort_values("date").iterrows():
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
    return S, avg, latest

def predict(S,avg,home,away,mg=10):
    lh=avg*S.loc[home,"attack"]*S.loc[away,"defense"]
    la=avg*S.loc[away,"attack"]*S.loc[home,"defense"]
    m=np.outer(poisson.pmf(range(mg),lh),poisson.pmf(range(mg),la))
    ph,pd_,pa=np.tril(m,-1).sum(),np.trace(m),np.triu(m,1).sum()
    met=np.outer(poisson.pmf(range(mg),lh/3),poisson.pmf(range(mg),la/3))
    phe,pde,pae=np.tril(met,-1).sum(),np.trace(met),np.triu(met,1).sum()
    return lh,la,ph,pd_,pa,pd_,pd_*pde,ph+pd_*(phe+pde*0.5),pa+pd_*(pae+pde*0.5)


@st.cache_data(ttl=3600)
def get_form(league, team, n=5):
    df = get_data(league)
    games = df[(df["home_team"]==team)|(df["away_team"]==team)].sort_values("date").tail(n)
    form=[]
    for _,r in games.iterrows():
        if r["home_team"]==team: gf,ga=r["home_score"],r["away_score"]
        else: gf,ga=r["away_score"],r["home_score"]
        form.append("W" if gf>ga else ("L" if gf<ga else "D"))
    return form

def top_scoreline(avg,S,home,away,mg=8):
    lh=avg*S.loc[home,"attack"]*S.loc[away,"defense"]
    la=avg*S.loc[away,"attack"]*S.loc[home,"defense"]
    best=(0,0,0.0)
    for i in range(mg):
        for j in range(mg):
            p=poisson.pmf(i,lh)*poisson.pmf(j,la)
            if p>best[2]: best=(i,j,p)
    return best

def slabel(tr,val,d=False):
    v=(1/val) if d else val
    return tr["strong"] if v>=1.15 else (tr["weak"] if v<=0.85 else tr["avg"])

# ==================== FLAGS ====================
SPECIAL_ISO={"England":"gb-eng","Scotland":"gb-sct","Wales":"gb-wls","Northern Ireland":"gb-nir","Kosovo":"xk"}
@st.cache_data
def flag_url(name):
    if name in SPECIAL_ISO: return f"https://flagcdn.com/w80/{SPECIAL_ISO[name]}.png"
    try:
        iso2=coco.convert(names=name,to="ISO2")
        if iso2 and iso2!="not found" and len(iso2)==2:
            return f"https://flagcdn.com/w80/{iso2.lower()}.png"
    except Exception: pass
    return None
def flag_img(name,h=22):
    u=flag_url(name)
    return f"<img src='{u}' height='{h}' style='vertical-align:middle;border-radius:3px;margin:0 4px;'>" if u else "⚽"


# ==================== CLUB BADGES (for leagues) ====================
CLUB_SEARCH = {
    "Athletic Club":"Athletic Bilbao","CA Osasuna":"Osasuna","CD Leganés":"Leganes",
    "Club Atlético de Madrid":"Atletico Madrid","Deportivo Alavés":"Alaves","Elche CF":"Elche",
    "FC Barcelona":"Barcelona","Getafe CF":"Getafe","Girona FC":"Girona","Levante UD":"Levante",
    "RC Celta de Vigo":"Celta Vigo","RCD Espanyol de Barcelona":"Espanyol","RCD Mallorca":"Mallorca",
    "Rayo Vallecano de Madrid":"Rayo Vallecano","Real Betis Balompié":"Real Betis",
    "Real Madrid CF":"Real Madrid","Real Oviedo":"Real Oviedo","Real Sociedad de Fútbol":"Real Sociedad",
    "Real Valladolid CF":"Real Valladolid","Sevilla FC":"Sevilla","UD Las Palmas":"Las Palmas",
    "Valencia CF":"Valencia","Villarreal CF":"Villarreal",
    "AFC Bournemouth":"Bournemouth","Arsenal FC":"Arsenal","Aston Villa FC":"Aston Villa",
    "Brentford FC":"Brentford","Brighton & Hove Albion FC":"Brighton","Burnley FC":"Burnley",
    "Chelsea FC":"Chelsea","Crystal Palace FC":"Crystal Palace","Everton FC":"Everton",
    "Fulham FC":"Fulham","Ipswich Town FC":"Ipswich","Leeds United FC":"Leeds",
    "Leicester City FC":"Leicester","Liverpool FC":"Liverpool","Manchester City FC":"Manchester City",
    "Manchester United FC":"Manchester United","Newcastle United FC":"Newcastle",
    "Nottingham Forest FC":"Nottingham Forest","Southampton FC":"Southampton",
    "Sunderland AFC":"Sunderland","Tottenham Hotspur FC":"Tottenham","West Ham United FC":"West Ham",
    "Wolverhampton Wanderers FC":"Wolverhampton",
    "1. FC Heidenheim 1846":"Heidenheim","1. FC Köln":"FC Koln","1. FC Union Berlin":"Union Berlin",
    "1. FSV Mainz 05":"Mainz","Bayer 04 Leverkusen":"Bayer Leverkusen","Borussia Dortmund":"Borussia Dortmund",
    "Borussia Mönchengladbach":"Borussia Monchengladbach","Eintracht Frankfurt":"Eintracht Frankfurt",
    "FC Augsburg":"Augsburg","FC Bayern München":"Bayern Munich","FC St. Pauli 1910":"St Pauli",
    "Hamburger SV":"Hamburg","RB Leipzig":"RB Leipzig","SC Freiburg":"Freiburg","SV Werder Bremen":"Werder Bremen",
    "TSG 1899 Hoffenheim":"Hoffenheim","VfB Stuttgart":"VfB Stuttgart","VfL Wolfsburg":"Wolfsburg",
}
@st.cache_data(ttl=86400, show_spinner=False)
def get_badge(team_name):
    import urllib.parse
    search = CLUB_SEARCH.get(team_name, team_name)
    try:
        q = urllib.parse.quote(search)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={q}"
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        data = json.load(urllib.request.urlopen(req, timeout=8))
        if data.get('teams'):
            for t in data['teams']:
                if t.get('strSport')=='Soccer':
                    b = t.get('strBadge')
                    if b: return b + "/tiny"
    except Exception:
        pass
    return None
def badge_img(name, h=22):
    u = get_badge(name)
    return f"<img src='{u}' height='{h}' style='vertical-align:middle;margin:0 4px;'>" if u else "⚽"

# ==================== ARABIC NAMES (nations) ====================
AR_NAMES={"مصر":"Egypt","الأرجنتين":"Argentina","البرازيل":"Brazil","السعودية":"Saudi Arabia",
 "المغرب":"Morocco","الجزائر":"Algeria","تونس":"Tunisia","قطر":"Qatar","الإمارات":"United Arab Emirates",
 "العراق":"Iraq","الأردن":"Jordan","الكويت":"Kuwait","عمان":"Oman","البحرين":"Bahrain","لبنان":"Lebanon",
 "سوريا":"Syria","فلسطين":"Palestine","ليبيا":"Libya","السودان":"Sudan","اليمن":"Yemen","فرنسا":"France",
 "ألمانيا":"Germany","إسبانيا":"Spain","البرتغال":"Portugal","إنجلترا":"England","إيطاليا":"Italy",
 "هولندا":"Netherlands","بلجيكا":"Belgium","كرواتيا":"Croatia","السنغال":"Senegal","نيجيريا":"Nigeria",
 "غانا":"Ghana","الكاميرون":"Cameroon","اليابان":"Japan","كوريا الجنوبية":"South Korea","أستراليا":"Australia",
 "أمريكا":"United States","المكسيك":"Mexico","الأوروجواي":"Uruguay","كولومبيا":"Colombia","إيران":"Iran"}
EN_TO_AR={v:k for k,v in AR_NAMES.items()}
# Arabic names for clubs
CLUB_AR = {
    "Athletic Club":"أتلتيك بيلباو","CA Osasuna":"أوساسونا","CD Leganés":"ليغانيس",
    "Club Atlético de Madrid":"أتلتيكو مدريد","Deportivo Alavés":"ألافيس","Elche CF":"إلتشي",
    "FC Barcelona":"برشلونة","Getafe CF":"خيتافي","Girona FC":"جيرونا","Levante UD":"ليفانتي",
    "RC Celta de Vigo":"سيلتا فيغو","RCD Espanyol de Barcelona":"إسبانيول","RCD Mallorca":"مايوركا",
    "Rayo Vallecano de Madrid":"رايو فاييكانو","Real Betis Balompié":"ريال بيتيس",
    "Real Madrid CF":"ريال مدريد","Real Oviedo":"ريال أوفييدو","Real Sociedad de Fútbol":"ريال سوسيداد",
    "Real Valladolid CF":"بلد الوليد","Sevilla FC":"إشبيلية","UD Las Palmas":"لاس بالماس",
    "Valencia CF":"فالنسيا","Villarreal CF":"فياريال",
    "AFC Bournemouth":"بورنموث","Arsenal FC":"آرسنال","Aston Villa FC":"أستون فيلا",
    "Brentford FC":"برينتفورد","Brighton & Hove Albion FC":"برايتون","Burnley FC":"بيرنلي",
    "Chelsea FC":"تشيلسي","Crystal Palace FC":"كريستال بالاس","Everton FC":"إيفرتون",
    "Fulham FC":"فولهام","Ipswich Town FC":"إيبسويتش","Leeds United FC":"ليدز يونايتد",
    "Leicester City FC":"ليستر سيتي","Liverpool FC":"ليفربول","Manchester City FC":"مانشستر سيتي",
    "Manchester United FC":"مانشستر يونايتد","Newcastle United FC":"نيوكاسل",
    "Nottingham Forest FC":"نوتينغهام فورست","Southampton FC":"ساوثهامبتون",
    "Sunderland AFC":"سندرلاند","Tottenham Hotspur FC":"توتنهام","West Ham United FC":"وست هام",
    "Wolverhampton Wanderers FC":"وولفرهامبتون",
    "1. FC Heidenheim 1846":"هايدنهايم","1. FC Köln":"كولن","1. FC Union Berlin":"يونيون برلين",
    "1. FSV Mainz 05":"ماينز","Bayer 04 Leverkusen":"باير ليفركوزن","Borussia Dortmund":"بوروسيا دورتموند",
    "Borussia Mönchengladbach":"مونشنغلادباخ","Eintracht Frankfurt":"آينتراخت فرانكفورت",
    "FC Augsburg":"أوغسبورغ","FC Bayern München":"بايرن ميونخ","FC St. Pauli 1910":"سانت باولي",
    "Hamburger SV":"هامبورغ","RB Leipzig":"لايبزيغ","SC Freiburg":"فرايبورغ","SV Werder Bremen":"فيردر بريمن",
    "TSG 1899 Hoffenheim":"هوفنهايم","VfB Stuttgart":"شتوتغارت","VfL Wolfsburg":"فولفسبورغ",
}

def disp(team,lang):
    if lang=="ar":
        if team in EN_TO_AR: return EN_TO_AR[team]
        if team in CLUB_AR: return CLUB_AR[team]
    return team

# ==================== TRANSLATIONS ====================
T={
 "en":{"title":"Match Predictor","subtitle":"Live football odds from thousands of real matches",
  "lang_label":"🌐 العربية","league":"Choose competition","home":"Team A","away":"Team B",
  "knockout":"Knockout match (extra time & penalties)","predict":"⚽ Show the odds",
  "loading":"Pulling latest data & training model...","pick":"⚠️ Please pick two different teams.",
  "win":"win","draw":"Draw","exp_goals":"Expected goals (avg)","in90":"📊 Win probability (90 minutes)",
  "decided90":"Decided in 90'","extratime":"Extra time","penalties":"Penalties",
  "knockout_head":"⏱️ If it's a knockout","qualify":"🏆 Who goes through","strengths":"💪 Team strengths",
  "attack":"Attack","defense":"Defense","elo":"Rating","strong":"strong","avg":"average","weak":"weak",
  "form":"📈 Recent form (last 5)","likely":"🎯 Most likely score","share":"📸 Share this result","download":"⬇️ Download image",
  "updated":"Data updated to","disclaimer":"These are odds from past form, not certainties. Just for fun ⚽",
  "verdict_fav":"**{a}** are the favourites, but **{b}** can cause an upset.",
  "verdict_close":"It's a coin-flip — **{a}** and **{b}** are very evenly matched!",
  "pens_note":"High chance of penalties 🥅 — anything can happen in a shootout!",
  "L_national":"National teams","L_laliga":"La Liga (ES)","L_epl":"Premier League (EN)","L_bundesliga":"Bundesliga (DE)"},
 "ar":{"title":"حاسبة فرص الماتشات","subtitle":"فرص حيّة محسوبة من آلاف الماتشات الحقيقية",
  "lang_label":"🌐 English","league":"اختار البطولة","home":"الفريق الأول","away":"الفريق التاني",
  "knockout":"ماتش خروج مغلوب (وقت إضافي وجزا)","predict":"⚽ وريني الفرص",
  "loading":"بنسحب آخر داتا وبندرّب الموديل...","pick":"⚠️ اختار فريقين مختلفين.",
  "win":"يكسب","draw":"تعادل","exp_goals":"متوسط الأهداف المتوقّع","in90":"📊 احتمالات الفوز (في 90 دقيقة)",
  "decided90":"يتحسم في 90'","extratime":"وقت إضافي","penalties":"ضربات جزا",
  "knockout_head":"⏱️ لو خروج مغلوب","qualify":"🏆 مين يتأهّل","strengths":"💪 قوة الفريقين",
  "attack":"الهجوم","defense":"الدفاع","elo":"التصنيف","strong":"قوي","avg":"متوسط","weak":"ضعيف",
  "form":"📈 آخر 5 نتايج","likely":"🎯 النتيجة الأرجح","share":"📸 شارك النتيجة","download":"⬇️ حمّل الصورة",
  "updated":"الداتا محدّثة حتى","disclaimer":"دي فرص محسوبة من الأداء السابق مش نتايج مؤكدة. للتسلية ⚽",
  "verdict_fav":"**{a}** هو المرشّح، بس **{b}** ممكن يعمل مفاجأة.",
  "verdict_close":"الماتش ونص — **{a}** و **{b}** متقاربين جداً!",
  "pens_note":"احتمال كبير لضربات الجزا 🥅 — وفيها أي حاجة ممكن تحصل!",
  "L_national":"المنتخبات","L_laliga":"الدوري الإسباني","L_epl":"الدوري الإنجليزي","L_bundesliga":"الدوري الألماني"},
}


# ==================== SHAREABLE RESULT CARD ====================
def build_card(home, away, ph, pd_, pa, si, sj, pen_pct, lang="en"):
    W,H=800,500
    img=Image.new("RGB",(W,H),(13,27,42)); d=ImageDraw.Draw(img)
    try: fontpath="Cairo.ttf"; ImageFont.truetype(fontpath,20)
    except Exception: fontpath=None
    def F(sz): return ImageFont.truetype(fontpath,sz) if fontpath else ImageFont.load_default()
    ar=(lang=="ar" and _AR_OK)
    def fix(t): return get_display(arabic_reshaper.reshape(t)) if ar else t
    def C(x,y,t,sz,col): d.text((x,y),fix(t),font=F(sz),fill=col,anchor="ma")
    def L(x,y,t,sz,col): d.text((x,y),fix(t),font=F(sz),fill=col,anchor="la")
    def R(x,y,t,sz,col): d.text((x,y),fix(t),font=F(sz),fill=col,anchor="ra")
    def Craw(x,y,t,sz,col): d.text((x,y),t,font=F(sz),fill=col,anchor="ma")
    d.rectangle([0,0,W,6],fill=(0,184,148))
    C(W//2,28,"⚽ Match Predictor",26,(232,238,245))
    C(200,88,home[:18],26,(232,238,245)); Craw(W//2,92,"VS",28,(120,120,130)); C(600,88,away[:18],26,(232,238,245))
    Craw(W//2,140,f"{si}  -  {sj}",64,(255,255,255))
    C(W//2,225,"النتيجة الأرجح" if lang=="ar" else "Most likely score",18,(150,155,165))
    bx,by,bw,bh=80,285,640,46
    sh=int(bw*ph); sd=int(bw*pd_); sa=bw-sh-sd
    d.rectangle([bx,by,bx+sh,by+bh],fill=(214,48,49))
    d.rectangle([bx+sh,by,bx+sh+sd,by+bh],fill=(99,110,114))
    d.rectangle([bx+sh+sd,by,bx+bw,by+bh],fill=(9,132,227))
    if sh>44: d.text((bx+sh//2,by+bh//2),f"{ph*100:.0f}%",font=F(20),fill=(255,255,255),anchor="mm")
    if sd>44: d.text((bx+sh+sd//2,by+bh//2),f"{pd_*100:.0f}%",font=F(20),fill=(255,255,255),anchor="mm")
    if sa>44: d.text((bx+sh+sd+sa//2,by+bh//2),f"{pa*100:.0f}%",font=F(20),fill=(255,255,255),anchor="mm")
    win="يكسب" if lang=="ar" else "win"; draw="تعادل" if lang=="ar" else "Draw"
    L(bx,by+bh+14,f"{home[:12]} {win}",16,(180,185,195))
    C(W//2,by+bh+14,draw,16,(180,185,195))
    R(bx+bw,by+bh+14,f"{away[:12]} {win}",16,(180,185,195))
    if pen_pct>0:
        pen=(f"احتمال ضربات جزا {pen_pct*100:.0f}%" if lang=="ar" else f"{pen_pct*100:.0f}% chance of penalties")
        C(W//2,406,"🥅 "+pen,20,(0,206,201))
    C(W//2,458,"نموذج Poisson + Elo · للتسلية" if lang=="ar" else "Poisson + Elo model  ·  for fun",14,(120,125,135))
    buf=io.BytesIO(); img.save(buf,format="PNG"); buf.seek(0)
    return buf

# ==================== STYLING ====================
st.markdown("""<style>
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.stApp{background:linear-gradient(160deg,#0d1b2a 0%,#1b263b 100%);}
h1,h2,h3,h4,p,label,span,div{color:#e8eef5 !important;}
.block-container{padding-top:1.2rem;max-width:760px;}
[data-testid="stMetricValue"]{font-size:2rem !important;font-weight:800 !important;}
.stButton>button{background:linear-gradient(90deg,#00b894,#00cec9);color:#003;font-weight:800;
 border:none;border-radius:12px;padding:.6rem 1rem;font-size:1.05rem;}
.stButton>button:hover{filter:brightness(1.1);}
.hero{text-align:center;padding:4px 0 2px;}
.matchup{display:flex;align-items:center;justify-content:center;gap:14px;margin:14px 0;}
.team-card{background:rgba(255,255,255,.06);border-radius:14px;padding:12px 18px;text-align:center;min-width:120px;}
.team-name{font-weight:700;margin-top:4px;font-size:1rem;}
.vs{font-size:1.3rem;font-weight:800;opacity:.6;}
.verdict{background:rgba(255,255,255,.07);border-left:4px solid #00b894;padding:14px 18px;
 border-radius:10px;font-size:1.1rem;margin:10px 0;}
.bar-wrap{display:flex;height:38px;border-radius:10px;overflow:hidden;margin:8px 0 4px;}
.bar-seg{display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.95rem;color:#fff;}
.form-badge{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;border-radius:6px;font-weight:800;font-size:.8rem;margin:0 2px;color:#fff;}
.fw{background:#00b894;} .fd{background:#636e72;} .fl{background:#d63031;}

/* --- Dropdown menu fix: dark background + visible text --- */
div[data-baseweb="select"] > div{
  background-color:#1b263b !important;
  border-color:rgba(255,255,255,.2) !important;
}
div[data-baseweb="select"] *{ color:#e8eef5 !important; }
/* the popover list that opens */
ul[role="listbox"], div[data-baseweb="popover"] div[data-baseweb="menu"]{
  background-color:#1b263b !important;
}
ul[role="listbox"] li, li[role="option"]{
  background-color:#1b263b !important;
  color:#e8eef5 !important;
}
ul[role="listbox"] li:hover, li[role="option"]:hover{
  background-color:#2a3b52 !important;
}
/* radio buttons text */
div[role="radiogroup"] label{ color:#e8eef5 !important; }
</style>""", unsafe_allow_html=True)

# ==================== UI ====================
if "lang" not in st.session_state: st.session_state.lang="en"
tr=T[st.session_state.lang]
lang=st.session_state.lang
if lang=="ar":
    st.markdown('<style>.main *{direction:rtl;text-align:right;}</style>',unsafe_allow_html=True)

# hero icons
st.markdown("<div class='hero'><div style='font-size:2.4rem;'>🏆⚽</div></div>",unsafe_allow_html=True)
# language toggle centered under the icons
lc1,lc2,lc3=st.columns([1,1,1])
with lc2:
    if st.button(tr["lang_label"], use_container_width=True):
        st.session_state.lang="ar" if st.session_state.lang=="en" else "en"; st.rerun()
tr=T[st.session_state.lang]; lang=st.session_state.lang
# title + subtitle
st.markdown(f"<div class='hero'><h1 style='margin:0;'>{tr['title']}</h1>"
            f"<p style='opacity:.7;margin-top:2px;'>{tr['subtitle']}</p></div>",unsafe_allow_html=True)

# league picker
league_opts={"national":tr["L_national"],"laliga":tr["L_laliga"],"epl":tr["L_epl"],"bundesliga":tr["L_bundesliga"]}
league=st.radio(tr["league"],list(league_opts.keys()),
                format_func=lambda k:league_opts[k],horizontal=True)

is_national = (league=="national")

with st.spinner(tr["loading"]):
    S,avg,latest=train(league)
teams=sorted(S.index.tolist())

di=teams.index("Egypt") if (is_national and "Egypt" in teams) else 0
ai=teams.index("Argentina") if (is_national and "Argentina" in teams) else 1
col1,col2=st.columns(2)
with col1:
    home=st.selectbox(tr["home"],teams,index=di,format_func=lambda t:disp(t,lang))
with col2:
    away=st.selectbox(tr["away"],teams,index=ai,format_func=lambda t:disp(t,lang))

# knockout option only for national teams (leagues have no penalties)
knockout = st.checkbox(tr["knockout"],value=True) if is_national else False

# nations get country flags; clubs get team badges
def team_flag(t,h=22):
    return flag_img(t,h) if is_national else badge_img(t,h)

fh,fa=team_flag(home,54),team_flag(away,54)
fh_s,fa_s=team_flag(home,22),team_flag(away,22)
st.markdown(f"<div class='matchup'>"
    f"<div class='team-card'><div>{fh}</div><div class='team-name'>{disp(home,lang)}</div></div>"
    f"<div class='vs'>VS</div>"
    f"<div class='team-card'><div>{fa}</div><div class='team-name'>{disp(away,lang)}</div></div>"
    f"</div>",unsafe_allow_html=True)

if st.button(tr["predict"],type="primary",use_container_width=True):
    if home==away:
        st.warning(tr["pick"])
    else:
        hn,an=disp(home,lang),disp(away,lang)
        lh,la,ph,pd_,pa,g_et,g_pen,fin_h,fin_a=predict(S,avg,home,away)
        diff=abs(ph-pa)
        if diff<0.08: verdict=tr["verdict_close"].format(a=f"{fh_s} {hn}",b=f"{fa_s} {an}")
        elif ph>pa: verdict=tr["verdict_fav"].format(a=f"{fh_s} {hn}",b=f"{fa_s} {an}")
        else: verdict=tr["verdict_fav"].format(a=f"{fa_s} {an}",b=f"{fh_s} {hn}")
        st.markdown(f"<div class='verdict'>{verdict}</div>",unsafe_allow_html=True)

        st.markdown(f"#### {tr['in90']}")
        st.markdown(f"<div class='bar-wrap'>"
            f"<div class='bar-seg' style='width:{ph*100:.0f}%;background:#d63031;'>{ph*100:.0f}%</div>"
            f"<div class='bar-seg' style='width:{pd_*100:.0f}%;background:#636e72;'>{pd_*100:.0f}%</div>"
            f"<div class='bar-seg' style='width:{pa*100:.0f}%;background:#0984e3;'>{pa*100:.0f}%</div></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:.9rem;opacity:.85;'>"
            f"<span>{fh_s} {hn} {tr['win']}</span><span>{tr['draw']}</span><span>{an} {tr['win']} {fa_s}</span></div>",
            unsafe_allow_html=True)

        # most likely scoreline
        si,sj,sp=top_scoreline(avg,S,home,away)
        st.markdown(f"<div style='text-align:center;margin:14px 0;'>"
            f"<span style='opacity:.7;'>{tr['likely']}:</span> "
            f"<span style='font-size:1.4rem;font-weight:800;'>{hn} {si} - {sj} {an}</span> "
            f"<span style='opacity:.6;'>({sp*100:.0f}%)</span></div>",unsafe_allow_html=True)

        g1,g2=st.columns(2)
        g1.metric(f"{hn} · {tr['exp_goals']}",f"{lh:.2f}")
        g2.metric(f"{an} · {tr['exp_goals']}",f"{la:.2f}")

        if knockout:
            st.markdown(f"#### {tr['knockout_head']}")
            k1,k2,k3=st.columns(3)
            k1.metric(tr["decided90"],f"{(ph+pa)*100:.0f}%")
            k2.metric(tr["extratime"],f"{g_et*100:.0f}%")
            k3.metric(tr["penalties"],f"{g_pen*100:.0f}%")
            if g_pen>0.15: st.info(tr["pens_note"])
            st.markdown(f"#### {tr['qualify']}")
            q1,q2=st.columns(2)
            q1.metric(f"{hn}",f"{fin_h*100:.0f}%")
            q2.metric(f"{an}",f"{fin_a*100:.0f}%")

        st.markdown(f"#### {tr['strengths']}")
        for team in [home,away]:
            atk=S.loc[team,"attack"]; dfn=S.loc[team,"defense"]; el=int(S.loc[team,"elo"])
            st.markdown(f"<div style='background:rgba(255,255,255,.05);padding:10px 14px;border-radius:10px;margin:5px 0;'>"
                f"<b>{team_flag(team,20)} {disp(team,lang)}</b> &nbsp;·&nbsp; {tr['attack']}: <b>{atk:.2f}</b> ({slabel(tr,atk)}) "
                f"&nbsp;·&nbsp; {tr['defense']}: <b>{dfn:.2f}</b> ({slabel(tr,dfn,True)}) "
                f"&nbsp;·&nbsp; {tr['elo']}: <b>{el}</b></div>",unsafe_allow_html=True)

        # recent form
        st.markdown(f"#### {tr['form']}")
        def form_html(team):
            f=get_form(league,team)
            cls={"W":"fw","D":"fd","L":"fl"}
            badges="".join(f"<span class='form-badge {cls[x]}'>{x}</span>" for x in f)
            return badges if badges else "<span style='opacity:.5;'>—</span>"
        for team in [home,away]:
            st.markdown(f"<div style='margin:6px 0;'><b>{team_flag(team,20)} {disp(team,lang)}</b> &nbsp; {form_html(team)}</div>",
                        unsafe_allow_html=True)

        # shareable image card
        st.markdown(f"#### {tr['share']}")
        try:
            card=build_card(home,away,ph,pd_,pa,si,sj,(g_pen if knockout else 0),"en")
            st.image(card, use_container_width=True)
            st.download_button(tr['download'], data=card.getvalue(),
                file_name=f"{home}_vs_{away}.png", mime="image/png", use_container_width=True)
        except Exception as e:
            st.caption("Image card unavailable")

        st.divider()
        st.caption(f"📅 {tr['updated']} {latest.date()}  ·  {tr['disclaimer']}")
