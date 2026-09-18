import os
import io
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Detector de Tumores Cerebrales", layout="centered")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NOMBRES = {
    "glioma": "Glioma", 
    "meningioma": "Meningioma",
    "notumor": "Sin tumor", 
    "pituitary": "Tumor pituitario",
}


MODELOS = {
    "CNN tradicional": {"ruta": "modelo/cnn.keras",
                        "acc": "95.86 %",
                        "params": "3.45 M",
                        "enfoque": "Entrenada desde cero"},
    
    "ResNet50": {"ruta": "modelo/resnet.keras",
                 "acc": "93.45 %",
                 "params": "23.5 M",
                 "enfoque": "Transfer learning"},
    
    "EfficientNetB0": {"ruta": "modelo/efficientnet.keras",
                       "acc": "89.77 %",
                       "params": "4.05 M",
                       "enfoque": "Transfer learning"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp { background:#ffffff; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none; }
header[data-testid="stHeader"] { background:transparent; height:0; }
.block-container { max-width:760px; padding-top:2.6rem; padding-bottom:4rem; }

html, body, [class*="css"], .stMarkdown { font-family:'Source Sans 3', system-ui, sans-serif; }

.dt-title { font-size:27px; font-weight:700; letter-spacing:-.01em; margin:0 0 5px; color:#14181c; }
.dt-lede  { font-size:15px; color:#5f6873; margin:0 0 30px; }
.dt-label { font-size:13px; color:#98a0a8; margin:2px 0 10px; }

.dt-fname { font-family:'IBM Plex Mono', monospace; font-size:13.5px; color:#14181c; word-break:break-all; }
.dt-fdesc { font-size:13px; color:#5f6873; margin-top:3px; }

/* pestañas */
.stTabs [data-baseweb="tab-list"] { gap:26px; border-bottom:1px solid #e7e9ec; }
.stTabs [data-baseweb="tab"] { padding:0 1px 10px; }
.stTabs [data-baseweb="tab"] p { font-size:15px; font-weight:500; color:#5f6873; }
.stTabs [aria-selected="true"] p { color:#14181c; font-weight:600; }

/* metadatos del modelo */
.dt-meta { display:flex; flex-wrap:wrap; gap:6px 40px; padding:22px 0; margin-bottom:22px;
           border-bottom:1px solid #e7e9ec; }
.dt-meta > div { display:flex; flex-direction:column; gap:2px; }
.dt-meta .lbl { font-size:12.5px; color:#98a0a8; }
.dt-meta .val { font-family:'IBM Plex Mono', monospace; font-size:14px; color:#14181c;
                font-variant-numeric:tabular-nums; }

/* resultado */
.dt-result { display:flex; align-items:flex-end; justify-content:space-between; gap:20px;
             flex-wrap:wrap; margin-bottom:26px; }
.dt-result .who { font-size:13px; color:#98a0a8; margin:0 0 4px; }
.dt-result .cls { font-size:25px; font-weight:700; letter-spacing:-.01em; margin:0; color:#14181c; }
.dt-conf { text-align:right; }
.dt-conf .n { font-family:'IBM Plex Mono', monospace; font-size:27px; color:#14181c;
              font-variant-numeric:tabular-nums; line-height:1; }
.dt-conf .c { font-size:12.5px; color:#98a0a8; margin-top:5px; }

/* barras de probabilidad */
.dt-bars { display:flex; flex-direction:column; gap:14px; }
.dt-bar  { display:grid; grid-template-columns:132px 1fr 56px; align-items:center; gap:14px; }
.dt-bar .name { font-size:14px; color:#5f6873; }
.dt-bar.win .name { color:#14181c; font-weight:600; }
.dt-track { height:7px; border-radius:2px; background:#f0f1f3; overflow:hidden; }
.dt-fill  { height:100%; border-radius:2px; background:#dbdee2; }
.dt-bar.win .dt-fill { background:#1c2126; }
.dt-pct { font-family:'IBM Plex Mono', monospace; font-size:13px; text-align:right; color:#5f6873;
          font-variant-numeric:tabular-nums; }
.dt-bar.win .dt-pct { color:#14181c; }

.dt-foot { margin-top:34px; font-size:12.5px; color:#98a0a8; }

@media (max-width:520px){
  .dt-bar { grid-template-columns:96px 1fr 48px; gap:10px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo(ruta):
    return keras.models.load_model(ruta)

@st.cache_data(show_spinner=False)
def predecir(ruta, img_bytes):
    model = cargar_modelo(ruta)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    return model.predict(arr)[0]

def pct(x):
    return f"{x * 100:.1f}".replace(".", ",") + " %"

st.markdown('<p class="dt-title">Detector de Tumores Cerebrales</p>', unsafe_allow_html=True)
st.markdown('<p class="dt-lede">Clasificación de resonancias magnéticas cerebrales en cuatro categorías.</p>', unsafe_allow_html=True)

disponibles = {n: d for n, d in MODELOS.items() if os.path.exists(d["ruta"])}
if not disponibles:
    st.error("No se encontró ningún modelo en la carpeta 'modelo/'.")
    st.stop()

# ---------- Carga de la imagen ----------
archivo = st.file_uploader("Imagen MRI (JPG o PNG)", type=["jpg", "jpeg", "png"])

if archivo is None:
    st.stop()

img_bytes = archivo.getvalue()
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
ancho, alto = img.size

# ---------- Imagen analizada ----------
st.markdown('<p class="dt-label">Imagen analizada</p>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 3.2], vertical_alignment="center")
with c1:
    st.image(img, width=104)
with c2:
    st.markdown(
        f'<div class="dt-fname">{archivo.name}</div>'
        f'<div class="dt-fdesc">{ancho} × {alto} px</div>',
        unsafe_allow_html=True,
    )

nombres = list(disponibles.keys())
for tab, nombre in zip(st.tabs(nombres), nombres):
    with tab:
        d = disponibles[nombre]
        pred = predecir(d["ruta"], img_bytes)
        idx = int(np.argmax(pred))

        filas = ""
        for i, c in enumerate(CLASSES):
            win = "win" if i == idx else ""
            filas += (
                f'<div class="dt-bar {win}"><span class="name">{NOMBRES[c]}</span>'
                f'<span class="dt-track"><span class="dt-fill" style="width:{pred[i]*100:.1f}%"></span></span>'
                f'<span class="dt-pct">{pct(pred[i])}</span></div>'
            )

        st.markdown(
            f'<div class="dt-meta">'
            f'<div><span class="lbl">Exactitud en prueba</span><span class="val">{d["acc"]}</span></div>'
            f'<div><span class="lbl">Parámetros</span><span class="val">{d["params"]}</span></div>'
            f'<div><span class="lbl">Enfoque</span><span class="val">{d["enfoque"]}</span></div>'
            f'</div>'
            f'<div class="dt-result">'
            f'<div><p class="who">Predicción — {nombre}</p>'
            f'<p class="cls">{NOMBRES[CLASSES[idx]]}</p></div>'
            f'<div class="dt-conf"><div class="n">{pct(pred[idx])}</div><div class="c">Confianza</div></div>'
            f'</div>'
            f'<p class="dt-label">Probabilidad por categoría</p>'
            f'<div class="dt-bars">{filas}</div>',
            unsafe_allow_html=True,
        )
