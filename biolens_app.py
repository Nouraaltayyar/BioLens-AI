import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as transforms
from PIL import Image
import base64, io, os
from openai import OpenAI

st.set_page_config(page_title="BioLens AI", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")


CLASS_NAMES = sorted([
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy",
])
NUM_CLASSES = 15

CARE = {
    "Pepper,_bell___Bacterial_spot":  {"tips":["Spray copper fungicide every 7 days","Avoid overhead irrigation","Remove infected leaves immediately"],"severity":"moderate"},
    "Pepper,_bell___healthy":         {"tips":["Plant looks great — keep up the good care","Water consistently and avoid waterlogging","Check regularly for early signs"],"severity":"none"},
    "Potato___Early_blight":          {"tips":["Apply chlorothalonil fungicide","Remove lower infected leaves","Ensure good airflow between plants"],"severity":"moderate"},
    "Potato___Late_blight":           {"tips":["Act immediately — this is serious","Apply metalaxyl fungicide right away","Remove and destroy all infected plants"],"severity":"critical"},
    "Potato___healthy":               {"tips":["Plant looks great — keep up the good care","Water consistently and avoid waterlogging","Check regularly for early signs"],"severity":"none"},
    "Tomato___Bacterial_spot":        {"tips":["Spray copper fungicide every 7–10 days","Use drip irrigation instead of overhead","Rotate crops each season"],"severity":"moderate"},
    "Tomato___Early_blight":          {"tips":["Apply fungicide at first sign","Remove lower infected leaves","Add balanced fertilizer"],"severity":"moderate"},
    "Tomato___Late_blight":           {"tips":["Act immediately — this is serious","Apply metalaxyl fungicide right away","Reduce humidity in the growing area"],"severity":"critical"},
    "Tomato___Leaf_Mold":             {"tips":["Improve ventilation in greenhouse","Keep humidity below 85%","Apply an appropriate fungicide"],"severity":"moderate"},
    "Tomato___Septoria_leaf_spot":    {"tips":["Remove infected leaves promptly","Apply fungicide as directed","Avoid wetting the foliage when watering"],"severity":"moderate"},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"tips":["Apply miticide or neem oil","Spray water on the undersides of leaves","Introduce predatory mites as a natural control"],"severity":"moderate"},
    "Tomato___Target_Spot":           {"tips":["Apply an appropriate fungicide","Remove all infected leaves","Practice crop rotation"],"severity":"moderate"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":{"tips":["No cure — remove infected plants immediately","Control whitefly population with insecticide","Use virus-resistant varieties next season"],"severity":"critical"},
    "Tomato___Tomato_mosaic_virus":   {"tips":["No cure — remove infected plants","Sterilize all tools with bleach after use","Avoid touching plants after handling tobacco"],"severity":"high"},
    "Tomato___healthy":               {"tips":["Plant looks great — keep up the good care","Water consistently and avoid waterlogging","Check regularly for early signs"],"severity":"none"},
}
SEV_COLOR = {"critical":"#e74c3c","high":"#e67e22","moderate":"#f59e0b","low":"#5aad48","none":"#5aad48"}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

*, html, body, [class*="css"] { font-family:"DM Sans",sans-serif; box-sizing:border-box; }
.stApp, .main { background:#060f07 !important; color:#c8e6c0; }
section[data-testid="stSidebar"] { display:none !important; }
header, #MainMenu, footer { visibility:hidden !important; }
.block-container { padding: 0 !important; max-width:100% !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a1f0d; }
::-webkit-scrollbar-thumb { background:#2d6e20; border-radius:99px; }

/* Buttons */
div.stButton > button {
    border-radius:12px !important; font-weight:800 !important;
    font-size:15px !important; padding:14px 28px !important;
    transition:all .25s cubic-bezier(.4,0,.2,1) !important;
    font-family:"DM Sans",sans-serif !important; letter-spacing:.01em !important;
}
.btn-primary > button {
    background:linear-gradient(135deg,#5aad48,#3d8a2e) !important;
    color:#fff !important; border:none !important;
    box-shadow:0 4px 24px rgba(90,173,72,0.4) !important;
}
.btn-primary > button:hover {
    transform:translateY(-2px) scale(1.01) !important;
    box-shadow:0 8px 32px rgba(90,173,72,0.55) !important;
}
.btn-ghost > button {
    background:rgba(90,173,72,0.07) !important;
    color:#5aad48 !important;
    border:1.5px solid rgba(90,173,72,0.35) !important;
}
.btn-ghost > button:hover {
    background:rgba(90,173,72,0.14) !important;
    transform:translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background:rgba(90,173,72,0.04) !important;
    border:2px dashed rgba(90,173,72,0.3) !important;
    border-radius:16px !important; padding:4px !important;
    transition:border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(90,173,72,0.6) !important; }

/* Progress */
[data-testid="stProgressBar"] > div > div {
    background:linear-gradient(90deg,#3d8a2e,#5aad48,#8bc34a) !important;
    border-radius:99px !important;
}
[data-testid="stProgressBar"] { border-radius:99px !important; background:rgba(255,255,255,0.05) !important; height:8px !important; }

/* Spinner */
.stSpinner > div { border-top-color:#5aad48 !important; }

/* Warning */
.stAlert { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session ───────────────────────────────────────────────────────────────────
for k,v in [("page","onboarding"),("analyzed",False),("mode",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Icons ─────────────────────────────────────────────────────────────────────
ICONS = {
    "leaf":   '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    "scan":   '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" y1="12" x2="17" y2="12"/></svg>',
    "check":  '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "alert":  '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "chart":  '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "bulb":   '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "zap":    '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "cross":  '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    "globe":  '<svg xmlns="http://www.w3.org/2000/svg" width="SZ" height="SZ" viewBox="0 0 24 24" fill="none" stroke="CL" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
}
def ic(name, color="#5aad48", size=18):
    s = ICONS.get(name,"")
    return f'<span style="color:{color};vertical-align:middle;display:inline-flex;align-items:center;">{s.replace("SZ",str(size)).replace("CL",color)}</span>'

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="/content/efficientnet_b0_best.pth"):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Sequential(nn.Dropout(p=0.4,inplace=True), nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
def predict(img_pil, top_k=3):
    t = val_tf(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad(): probs = F.softmax(load_model()(t),dim=1)[0]
    top_p,top_i = probs.topk(top_k)
    return [(CLASS_NAMES[i],p.item()) for i,p in zip(top_i,top_p)]

# ── VLM ───────────────────────────────────────────────────────────────────────
def img_to_b64(img_pil, max_size=512):
    img=img_pil.copy()
    if max(img.size)>max_size: img.thumbnail((max_size,max_size))
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

def validate_and_route(img_pil):
    try:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"] 
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        prompt = ("Look at this image.\n1. Is it a plant leaf? YES or NO\n"
                  "2. If YES: is it tomato, potato, or pepper? Or other?\n"
                  "3. One sentence describing what you see.\n\n"
                  "Reply EXACTLY:\nIS_PLANT: YES or NO\nCROP: tomato or potato or pepper or other\nDESC: one sentence")
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_to_b64(img_pil)}},
                {"type":"text","text":prompt}]}],
            max_tokens=120)
        text = resp.choices[0].message.content.strip()
        is_plant,crop,desc = False,"other",""
        for line in text.splitlines():
            if ":" not in line: continue
            k,v = line.split(":",1); k=k.strip().upper(); v=v.strip()
            if k=="IS_PLANT": is_plant="YES" in v.upper()
            elif k=="CROP":   crop=v.lower()
            elif k=="DESC":   desc=v.capitalize()
        if not is_plant: return "not_plant",crop,desc
        if any(c in crop for c in ["tomato","potato","pepper"]): return "dataset_crop",crop,desc
        return "other_plant",crop,desc
    except Exception as e:
        return "dataset_crop","unknown",str(e)

def gpt_analyze_other(img_pil):
    try:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"] 
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        prompt = ("Analyze this plant leaf. Reply EXACTLY:\nPLANT: common name\nSTATUS: Healthy or Diseased\n"
                  "DISEASE: disease name or None\nSEVERITY: low or moderate or high or critical\n"
                  "TIP1: first tip\nTIP2: second tip\nTIP3: third tip")
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_to_b64(img_pil)}},
                {"type":"text","text":prompt}]}],
            max_tokens=220)
        text=resp.choices[0].message.content.strip()
        r={"plant":"Unknown","status":"Unknown","disease":"None","severity":"moderate","tips":[]}
        for line in text.splitlines():
            if ":" not in line: continue
            k,v=line.split(":",1); k=k.strip().upper(); v=v.strip()
            if k=="PLANT": r["plant"]=v
            elif k=="STATUS": r["status"]=v
            elif k=="DISEASE": r["disease"]=v
            elif k=="SEVERITY": r["severity"]=v.lower()
            elif k in("TIP1","TIP2","TIP3") and v: r["tips"].append(v)
        return r
    except Exception as e:
        return {"plant":"Unknown","status":"Error","disease":"None","severity":"moderate","tips":[str(e)]}

# ═════════════════════════════════════════════════════════════
# ONBOARDING
# ═════════════════════════════════════════════════════════════
if st.session_state.page == "onboarding":

    st.markdown(f"""
    <div style="min-height:100vh;background:linear-gradient(160deg,#060f07 0%,#0a1f0d 50%,#0d2b10 100%);
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:60px 24px 40px;">

      <!-- Glow orb -->
      <div style="position:absolute;top:15%;left:50%;transform:translateX(-50%);
                  width:500px;height:500px;border-radius:50%;
                  background:radial-gradient(circle,rgba(90,173,72,0.12) 0%,transparent 70%);
                  pointer-events:none;"></div>

      <!-- Logo -->
      <div style="width:80px;height:80px;background:linear-gradient(135deg,#1a4d10,#2d7a1f);
                  border-radius:24px;display:flex;align-items:center;justify-content:center;
                  box-shadow:0 8px 32px rgba(90,173,72,0.3);margin-bottom:28px;">
          {ic("leaf","#7dd56f",36)}
      </div>

      <!-- Title -->
      <div style="font-family:'Playfair Display',serif;font-size:clamp(42px,7vw,72px);
                  font-weight:900;color:#e8f5e2;line-height:1.05;text-align:center;margin-bottom:10px;">
          BioLens <span style="color:#5aad48;font-style:italic;">AI</span>
      </div>
      <div style="font-size:15px;color:#6a9a5e;text-align:center;margin-bottom:14px;letter-spacing:.03em;">
          Instant plant disease diagnosis
      </div>

      <!-- Divider -->
      <div style="width:48px;height:3px;background:linear-gradient(90deg,#5aad48,#8bc34a);
                  border-radius:99px;margin-bottom:36px;"></div>

      <!-- Stats -->
      <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:40px;max-width:600px;">
          <div style="background:rgba(255,255,255,0.03);backdrop-filter:blur(12px);
                      border:1px solid rgba(90,173,72,0.2);border-radius:18px;
                      padding:20px 28px;min-width:130px;text-align:center;
                      box-shadow:0 4px 24px rgba(0,0,0,0.3);">
              <div style="margin-bottom:8px;">{ic("target","#5aad48",24)}</div>
              <div style="font-family:'Playfair Display',serif;font-size:30px;font-weight:900;color:#5aad48;line-height:1;">99.4%</div>
              <div style="font-size:10px;color:#4a7a3a;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">Accuracy</div>
          </div>
          <div style="background:rgba(255,255,255,0.03);backdrop-filter:blur(12px);
                      border:1px solid rgba(90,173,72,0.2);border-radius:18px;
                      padding:20px 28px;min-width:130px;text-align:center;
                      box-shadow:0 4px 24px rgba(0,0,0,0.3);">
              <div style="margin-bottom:8px;">{ic("leaf","#5aad48",24)}</div>
              <div style="font-family:'Playfair Display',serif;font-size:30px;font-weight:900;color:#5aad48;line-height:1;">15</div>
              <div style="font-size:10px;color:#4a7a3a;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">Diseases</div>
          </div>
          <div style="background:rgba(255,255,255,0.03);backdrop-filter:blur(12px);
                      border:1px solid rgba(90,173,72,0.2);border-radius:18px;
                      padding:20px 28px;min-width:130px;text-align:center;
                      box-shadow:0 4px 24px rgba(0,0,0,0.3);">
              <div style="margin-bottom:8px;">{ic("zap","#5aad48",24)}</div>
              <div style="font-family:'Playfair Display',serif;font-size:30px;font-weight:900;color:#5aad48;line-height:1;">&lt;2s</div>
              <div style="font-size:10px;color:#4a7a3a;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">Analysis</div>
          </div>
          <div style="background:rgba(255,255,255,0.03);backdrop-filter:blur(12px);
                      border:1px solid rgba(90,173,72,0.2);border-radius:18px;
                      padding:20px 28px;min-width:130px;text-align:center;
                      box-shadow:0 4px 24px rgba(0,0,0,0.3);">
              <div style="margin-bottom:8px;">{ic("globe","#5aad48",24)}</div>
              <div style="font-family:'Playfair Display',serif;font-size:30px;font-weight:900;color:#5aad48;line-height:1;">Any</div>
              <div style="font-size:10px;color:#4a7a3a;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">Plant</div>
          </div>
      </div>

      <!-- Subtitle -->
      <p style="font-size:15px;color:#4a7a3a;max-width:420px;text-align:center;line-height:1.8;margin-bottom:36px;">
          Upload a leaf photo and get an instant diagnosis with personalised care recommendations.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,1.2,1])
    with c2:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("🌿  Start Analysis", use_container_width=True):
            st.session_state.page="analyze"; st.session_state.analyzed=False; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;padding:20px;font-size:12px;color:#2a4a22;">Developed 2026 · Agro-Mind Capstone · SDA AI Engineering</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# ANALYZE
# ═════════════════════════════════════════════════════════════
else:
    # Navbar
    st.markdown(f"""
    <div style="background:rgba(10,31,13,0.95);backdrop-filter:blur(20px);
                border-bottom:1px solid rgba(90,173,72,0.15);
                padding:16px 5%;display:flex;align-items:center;gap:12px;
                position:sticky;top:0;z-index:999;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#1a4d10,#2d7a1f);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;">
            {ic("leaf","#7dd56f",18)}
        </div>
        <span style="font-family:'Playfair Display',serif;font-size:18px;font-weight:900;color:#5aad48;">BioLens AI</span>
        <span style="margin-left:auto;font-size:11px;color:#3a6a2e;font-weight:600;
                     background:rgba(90,173,72,0.08);border:1px solid rgba(90,173,72,0.2);
                     border-radius:99px;padding:4px 12px;">
                    All Plants Supported
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 4% 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-bottom:36px;">
        <div style="font-family:'Playfair Display',serif;font-size:clamp(24px,4vw,38px);
                    font-weight:900;color:#e8f5e2;margin-bottom:8px;">Analyze Your Plant</div>
        <p style="font-size:13px;color:#4a7a3a;">Upload a leaf photo to receive an instant diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")

    # ── LEFT ──
    with left:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(90,173,72,0.12);
                    border-radius:20px;padding:24px;">
            <div style="font-size:13px;font-weight:700;color:#6a9a5e;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;
                        display:flex;align-items:center;gap:8px;">
                {ic("upload","#5aad48",14)} Upload Image
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            run = st.button("🔬  Analyze Plant", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if run:
                with st.spinner("Analyzing image..."):
                    mode, crop, desc = validate_and_route(img)
                if mode == "not_plant":
                    st.session_state.update({"analyzed":True,"mode":"rejected","crop":crop,"desc":desc})
                elif mode == "dataset_crop":
                    with st.spinner("Running diagnosis..."):
                        results = predict(img, top_k=3)
                    st.session_state.update({"analyzed":True,"mode":"efficientnet","results":results,"crop":crop,"desc":desc})
                else:
                    with st.spinner("Analyzing plant..."):
                        gpt_res = gpt_analyze_other(img)
                    st.session_state.update({"analyzed":True,"mode":"gpt_other","gpt_res":gpt_res,"crop":crop,"desc":desc})

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT ──
    with right:
        if not st.session_state.analyzed:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1.5px dashed rgba(90,173,72,0.2);
                        border-radius:20px;padding:64px 32px;text-align:center;">
                <div style="width:64px;height:64px;background:rgba(90,173,72,0.08);border-radius:16px;
                            display:flex;align-items:center;justify-content:center;margin:0 auto 16px;">
                    {ic("scan","#3a6a2e",28)}
                </div>
                <div style="font-size:15px;font-weight:600;color:#4a7a3a;margin-bottom:6px;">No image analyzed yet</div>
                <div style="font-size:13px;color:#2d5a22;">Upload a photo and click Analyze</div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.mode == "rejected":
            st.markdown(f"""
            <div style="background:rgba(231,76,60,0.06);border:1.5px solid rgba(231,76,60,0.25);
                        border-radius:20px;padding:40px 32px;text-align:center;">
                <div style="width:64px;height:64px;background:rgba(231,76,60,0.1);border-radius:16px;
                            display:flex;align-items:center;justify-content:center;margin:0 auto 16px;">
                    {ic("cross","#e74c3c",28)}
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:22px;font-weight:900;
                            color:#e74c3c;margin-bottom:8px;">Not a Plant Leaf</div>
                <div style="font-size:13px;color:#a07070;font-style:italic;max-width:280px;margin:0 auto;">
                    {st.session_state.desc}
                </div>
                <div style="margin-top:16px;font-size:12px;color:#6a4040;">
                    Please upload a clear plant leaf image
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.mode == "efficientnet":
            results  = st.session_state.results
            top_cls, top_conf = results[0]
            rec      = CARE.get(top_cls, {})
            tips     = rec.get("tips",["Consult an agricultural specialist"])
            severity = rec.get("severity","moderate")
            healthy  = "healthy" in top_cls.lower()
            sc       = "#5aad48" if healthy else SEV_COLOR.get(severity,"#e74c3c")
            sl       = "Healthy" if healthy else severity.capitalize()+" Risk"
            pp       = top_cls.split("___")
            pname    = pp[0].replace("_"," ")
            cond     = pp[1].replace("_"," ") if len(pp)>1 else ""
            bg       = "rgba(90,173,72,0.06)"  if healthy else "rgba(231,76,60,0.06)"
            bd       = "rgba(90,173,72,0.25)"  if healthy else "rgba(231,76,60,0.2)"
            icon_n   = "check" if healthy else "alert"

            # Main result card
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {bd};border-radius:20px;padding:28px;margin-bottom:20px;">
                <!-- Status pill -->
                <div style="display:inline-flex;align-items:center;gap:6px;
                            background:{sc}18;border:1px solid {sc}40;border-radius:99px;
                            padding:5px 14px;font-size:11px;font-weight:700;color:{sc};
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">
                    {ic(icon_n,sc,11)} {sl}
                </div>
                <!-- Plant name -->
                <div style="font-family:'Playfair Display',serif;font-size:26px;
                            font-weight:900;color:{sc};line-height:1.15;margin-bottom:4px;">{pname}</div>
                <div style="font-size:14px;color:#8ab08a;margin-bottom:20px;font-style:italic;">{cond}</div>
                <!-- Confidence meter -->
                <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:11px;font-weight:700;color:#6a9a5e;
                                text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">
                        <span>Confidence</span>
                        <span style="color:{sc};font-size:15px;">{top_conf*100:.1f}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:8px;overflow:hidden;">
                        <div style="width:{top_conf*100:.1f}%;height:100%;border-radius:99px;
                                    background:linear-gradient(90deg,{sc}99,{sc});"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if top_conf < 0.60:
                st.warning("⚠️ Low confidence — try a clearer, well-lit image with the leaf filling the frame.")

            # Top-3
            st.markdown(f"""
            <div style="font-size:12px;font-weight:700;color:#6a9a5e;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
                {ic("chart","#5aad48",13)} Top Predictions
            </div>
            """, unsafe_allow_html=True)
            for i,(cls,conf) in enumerate(results):
                label = cls.replace("___",": ").replace("_"," ")
                st.progress(conf, text=f"{'🥇🥈🥉'[i]}  {label}  —  {conf*100:.1f}%")

            # Tips
            tips_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;'
                f'border-bottom:1px solid rgba(90,173,72,0.08);">'
                f'<div style="min-width:20px;margin-top:1px;">{ic("check","#5aad48",13)}</div>'
                f'<span style="font-size:13px;color:#b8d8b0;line-height:1.6;">{t}</span></div>'
                for t in tips
            )
            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="font-size:12px;font-weight:700;color:#6a9a5e;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
                    {ic("bulb","#5aad48",13)} Care Recommendations
                </div>
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(90,173,72,0.12);
                            border-radius:16px;padding:4px 16px;">
                    {tips_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.mode == "gpt_other":
            gpt      = st.session_state.gpt_res
            healthy  = "healthy" in gpt["status"].lower()
            severity = gpt.get("severity","moderate")
            sc       = "#5aad48" if healthy else SEV_COLOR.get(severity,"#e74c3c")
            sl       = "Healthy" if healthy else severity.capitalize()+" Risk"
            icon_n   = "check" if healthy else "alert"
            bg       = "rgba(90,173,72,0.06)" if healthy else "rgba(231,76,60,0.06)"
            bd       = "rgba(90,173,72,0.25)" if healthy else "rgba(231,76,60,0.2)"
            disease_val = gpt.get("disease","")
            dis_html = ('<div style="font-size:14px;color:#8ab08a;margin-bottom:20px;font-style:italic;">'
                        + disease_val + '</div>') if disease_val.lower() not in("none","") else '<div style="margin-bottom:20px;"></div>'

            # "Being analyzed" notice — clean, no model name
            st.markdown(f"""
            <div style="background:rgba(90,173,72,0.05);border:1px solid rgba(90,173,72,0.15);
                        border-radius:12px;padding:10px 16px;font-size:12px;color:#4a7a3a;
                        margin-bottom:16px;display:flex;align-items:center;gap:8px;">
                {ic("globe","#4a7a3a",13)}
                <span>This plant is outside the primary dataset — analysis performed via AI vision</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {bd};border-radius:20px;padding:28px;margin-bottom:20px;">
                <div style="display:inline-flex;align-items:center;gap:6px;
                            background:{sc}18;border:1px solid {sc}40;border-radius:99px;
                            padding:5px 14px;font-size:11px;font-weight:700;color:{sc};
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">
                    {ic(icon_n,sc,11)} {sl}
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:26px;
                            font-weight:900;color:{sc};line-height:1.15;margin-bottom:4px;">{gpt["plant"]}</div>
                {dis_html}
            </div>
            """, unsafe_allow_html=True)

            if gpt.get("tips"):
                tips_html = "".join(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;'
                    f'border-bottom:1px solid rgba(90,173,72,0.08);">'
                    f'<div style="min-width:20px;margin-top:1px;">{ic("check","#5aad48",13)}</div>'
                    f'<span style="font-size:13px;color:#b8d8b0;line-height:1.6;">{t}</span></div>'
                    for t in gpt["tips"]
                )
                st.markdown(f"""
                <div style="margin-top:4px;">
                    <div style="font-size:12px;font-weight:700;color:#6a9a5e;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
                        {ic("bulb","#5aad48",13)} Care Recommendations
                    </div>
                    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(90,173,72,0.12);
                                border-radius:16px;padding:4px 16px;">
                        {tips_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Back button
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page="onboarding"; st.session_state.analyzed=False; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style="background:rgba(6,15,7,0.8);border-top:1px solid rgba(90,173,72,0.08);
                padding:20px 5%;text-align:center;margin-top:48px;">
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:4px;">
            {ic("leaf","#3d6e30",14)}
            <span style="color:#3d6e30;font-size:13px;font-weight:700;">BioLens AI</span>
        </div>
        <div style="font-size:11px;color:#2a4a22;">Developed 2026 · Agro-Mind Capstone · SDA AI Engineering</div>
    </div>
    """, unsafe_allow_html=True)
