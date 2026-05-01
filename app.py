"""
DenseNet121 Dental Caries Detection — Streamlit App (Enhanced)
Pages: Training Results | Predict & Grad-CAM | Model Architecture | Report Analysis
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# torch/torchvision/grad-cam lazy-loaded inside functions
# to avoid Windows DLL initialisation failure at Streamlit startup.
import streamlit as st
import numpy as np
import json
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "training_results")
WEIGHTS    = os.path.join(RESULT_DIR, "densenet121_caries_best.pth")
METRICS    = os.path.join(RESULT_DIR, "metrics.json")
HISTORY    = os.path.join(RESULT_DIR, "training_history.json")
CM_IMG     = os.path.join(RESULT_DIR, "confusion_matrix.png")
PDF_PATH   = os.path.join(BASE_DIR,   "DenseNet121_Caries_Report.pdf")

# ── Auto-download weights if missing ───────────────────────────────────────
WEIGHTS_URL = (
    "https://github.com/paru-is-bot/densenet121-dental-caries"
    "/releases/download/v1.0/densenet121_caries_best.pth"
)

def _ensure_weights():
    """Download model weights from GitHub Releases if not present locally."""
    if os.path.exists(WEIGHTS):
        return
    import urllib.request
    os.makedirs(RESULT_DIR, exist_ok=True)
    st.info("⬇️ Downloading model weights from GitHub Releases (~27 MB) …")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
        st.success("✅ Weights downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not download weights: {e}")
        st.stop()

CLASS_NAMES   = ["caries", "healthy"]
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_json(path):
    with open(path) as f: return json.load(f)

def load_cm():
    """Load confusion matrix from metrics.json (single source of truth)."""
    if os.path.exists(METRICS):
        data = load_json(METRICS)
        if "confusion_matrix" in data:
            return np.array(data["confusion_matrix"])
    # Fallback — matches confusion_matrix.png exactly
    return np.array([[96, 2], [8, 17]])

# ── model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    _ensure_weights()
    import torch
    import torch.nn as nn
    from torchvision import models
    device = _get_device()
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.classifier.in_features, 1))
    m.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    m.to(device).eval()
    for mod in m.modules():
        if hasattr(mod, "inplace"): mod.inplace = False
    return m

def preprocess(img):
    import torch
    from torchvision import transforms
    device = _get_device()
    t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return t(img).unsqueeze(0).to(device)

def generate_gradcam(model, tensor, raw_np):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    cam = GradCAM(model=model, target_layers=[model.features.norm5])
    gs  = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(0)])[0]
    return show_cam_on_image(raw_np, gs, use_rgb=True), gs

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="DenseNet121 — Dental Caries", page_icon="🦷", layout="wide")

st.markdown("""
<style>
#MainMenu,footer{visibility:hidden}
.metric-card{
  background:linear-gradient(135deg,#1e3a5f,#162032);
  border:1px solid #334155;border-radius:12px;padding:18px 22px;
  color:#e2e8f0;text-align:center}
.metric-card h3{margin:0 0 4px;font-size:12px;text-transform:uppercase;
  letter-spacing:1px;color:#64748b}
.metric-card p{margin:0;font-size:28px;font-weight:700}
.pred-caries{background:linear-gradient(135deg,#7f1d1d,#ef4444);
  border-radius:14px;padding:22px;color:#fff;text-align:center}
.pred-healthy{background:linear-gradient(135deg,#064e3b,#10b981);
  border-radius:14px;padding:22px;color:#fff;text-align:center}
.issue-card{border-radius:10px;padding:14px 18px;margin-bottom:10px}
</style>""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦷 DenseNet121")
    st.caption("Dental Caries Detection System")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Training Results",
        "🔬 Predict & Grad-CAM",
        "🏗️ Model Architecture",
        "📋 Report Analysis",
    ])
    st.markdown("---")
    st.caption("Device: `cpu` (loaded on first prediction)")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Training Results":
    st.title("📊 Training Results Dashboard")

    if os.path.exists(METRICS):
        m = load_json(METRICS)
        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,key,color) in zip([c1,c2,c3,c4],[
            ("Accuracy","accuracy","#34d399"),
            ("Precision","precision_weighted","#818cf8"),
            ("Recall","recall_weighted","#fbbf24"),
            ("F1 Score","f1_weighted","#fb923c"),
        ]):
            v = m.get(key,0)*100
            col.markdown(f"""<div class="metric-card">
                <h3>{lbl}</h3><p style="color:{color}">{v:.2f}%</p></div>""",
                unsafe_allow_html=True)
        st.markdown("")
        st.info(f"**Test Loss:** {m.get('test_loss','N/A')}  |  **Classes:** Caries / Healthy  |  **Backbone:** DenseNet121")

    st.markdown("---")

    if os.path.exists(HISTORY):
        h = load_json(HISTORY)
        ep = list(range(1, len(h["train_loss"])+1))

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("Loss — Train vs Val", "Accuracy — Train vs Val"))
        fig.add_trace(go.Scatter(x=ep, y=h["train_loss"], name="Train Loss",
            line=dict(color="#818cf8",width=2), mode="lines+markers", marker=dict(size=5)), 1,1)
        fig.add_trace(go.Scatter(x=ep, y=h["val_loss"], name="Val Loss",
            line=dict(color="#fb923c",width=2), mode="lines+markers", marker=dict(size=5)), 1,1)
        fig.add_trace(go.Scatter(x=ep, y=[a*100 for a in h["train_acc"]], name="Train Acc",
            line=dict(color="#34d399",width=2), mode="lines+markers", marker=dict(size=5)), 1,2)
        fig.add_trace(go.Scatter(x=ep, y=[a*100 for a in h["val_acc"]], name="Val Acc",
            line=dict(color="#fb7185",width=2), mode="lines+markers", marker=dict(size=5)), 1,2)
        fig.update_layout(height=400, template="plotly_dark",
            legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📈 Learning Rate Schedule"):
            lr_fig = go.Figure(go.Scatter(x=ep, y=h["lr"], mode="lines+markers",
                line=dict(color="#a78bfa", width=2)))
            lr_fig.update_layout(template="plotly_dark", height=280,
                xaxis_title="Epoch", yaxis_title="LR")
            st.plotly_chart(lr_fig, use_container_width=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Confusion Matrix")
        if os.path.exists(CM_IMG): st.image(CM_IMG, use_container_width=True)
    with cb:
        st.subheader("Per-Class Metrics")
        cm = load_cm()
        tp_c,fn_c,fp_c,tp_h = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
        pc = tp_c/(tp_c+fp_c); rc = tp_c/(tp_c+fn_c); f1c = 2*pc*rc/(pc+rc)
        ph = tp_h/(tp_h+fn_c); rh = tp_h/(tp_h+fp_c); f1h = 2*ph*rh/(ph+rh)
        bar = go.Figure()
        bar.add_trace(go.Bar(name="Caries", x=["Precision","Recall","F1"],
            y=[pc*100,rc*100,f1c*100], marker_color="#fb7185"))
        bar.add_trace(go.Bar(name="Healthy", x=["Precision","Recall","F1"],
            y=[ph*100,rh*100,f1h*100], marker_color="#34d399"))
        bar.update_layout(template="plotly_dark", height=320, barmode="group",
            yaxis=dict(range=[0,115]), legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT & GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict & Grad-CAM":
    st.title("🔬 Predict & Grad-CAM Visualisation")
    st.markdown("Upload a dental X-ray to classify it as **Caries** or **Healthy** "
                "and visualise where the model focuses via **Grad-CAM**.")
    st.markdown("---")

    uploaded = st.file_uploader("Choose an X-ray image",
        type=["png","jpg","jpeg","bmp","tiff"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        tensor  = preprocess(pil_img)
        raw_np  = np.array(pil_img.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255.

        with st.spinner("Loading model…"):
            model = load_model()

        import torch
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()

        if prob >= 0.5:
            label, conf, css = "Healthy", prob*100, "pred-healthy"
        else:
            label, conf, css = "Caries", (1-prob)*100, "pred-caries"

        st.markdown(f"""<div class="{css}">
            <h2 style="margin:0">Prediction: {label}</h2>
            <p style="font-size:20px;margin:6px 0 0">Confidence: {conf:.1f}%</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        with st.spinner("Generating Grad-CAM…"):
            cam_overlay, gs = generate_gradcam(model, tensor, raw_np)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)
        with c2:
            st.subheader("Grad-CAM Heatmap")
            fh, ah = plt.subplots(figsize=(4,4))
            ah.imshow(gs, cmap="jet"); ah.axis("off")
            plt.tight_layout(pad=0); st.pyplot(fh); plt.close(fh)
        with c3:
            st.subheader("Overlay")
            st.image(cam_overlay, use_container_width=True)

        st.markdown("---")
        st.caption("**Grad-CAM:** Red/warm = high model attention. Blue/cool = low influence.")
    else:
        st.info("👆 Upload a dental X-ray image to get started.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Model Architecture":
    st.title("🏗️ Model Architecture")
    st.markdown("Visual overview of the **DenseNet121** backbone with the custom "
                "binary classifier head used for dental caries detection.")
    st.markdown("---")

    # ── Architecture flow diagram ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120"); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)

    blocks = [
        ("Input\n224×224×3",   0.5,  "#334155",  "#94a3b8",  False),
        ("DenseBlock 1\n+ Transition",  2.1,  "#1e3a5f",  "#818cf8",  False),
        ("DenseBlock 2\n+ Transition",  3.7,  "#1e3a5f",  "#818cf8",  False),
        ("DenseBlock 3\n+ Transition",  5.3,  "#1e3a5f",  "#818cf8",  False),
        ("DenseBlock 4\n(Unfrozen)",    7.1,  "#1a3a2a",  "#34d399",  True),
        ("Norm5 BN\n(Unfrozen)",        8.9,  "#1a3a2a",  "#34d399",  True),
        ("Global Avg\nPool 1×1",        10.5, "#2d1b4e",  "#a78bfa",  False),
        ("Dropout\np=0.5",             11.9,  "#3b1f1f",  "#fb7185",  False),
        ("Linear\n1024→1  +  Sigmoid", 13.3,  "#1f2d1f",  "#4ade80",  False),
    ]

    bw, bh = 1.1, 1.8
    by = 1.6
    for (label, cx, fc, ec, unfrozen) in blocks:
        rect = mpatches.FancyBboxPatch((cx - bw/2, by), bw, bh,
            boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=2.0)
        ax.add_patch(rect)
        ax.text(cx, by + bh/2, label, ha="center", va="center",
                color=ec, fontsize=7.5, fontweight="bold", multialignment="center")
        if unfrozen:
            ax.text(cx, by - 0.28, "UNFROZEN", ha="center", color=ec,
                    fontsize=6.5, fontstyle="italic")

    # Arrows
    for i in range(len(blocks)-1):
        x0 = blocks[i][1] + bw/2
        x1 = blocks[i+1][1] - bw/2
        ax.annotate("", xy=(x1, by+bh/2), xytext=(x0, by+bh/2),
                    arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

    # Frozen badge
    ax.text(3.7, 4.0, "FROZEN (ImageNet weights locked)",
            ha="center", color="#64748b", fontsize=8, style="italic")
    ax.annotate("", xy=(5.3+bw/2, 4.0), xytext=(3.7+1.8, 4.0),
                arrowprops=dict(arrowstyle="-", color="#334155", lw=1))
    ax.annotate("", xy=(0.5-bw/2+0.1, 4.0), xytext=(3.7-1.8, 4.0),
                arrowprops=dict(arrowstyle="-", color="#334155", lw=1))

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── Architecture table ────────────────────────────────────────────────
    st.subheader("Configuration Details")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Layer Configuration**")
        rows = [
            ("Backbone",         "DenseNet121 — ImageNet pretrained"),
            ("Input",            "224 × 224 × 3 (RGB)"),
            ("Frozen Layers",    "DenseBlock 1–3 + all Transitions"),
            ("Unfrozen Block",   "DenseBlock 4  (16 dense layers)"),
            ("Unfrozen Norm",    "Norm5 (BatchNorm after DB4)"),
            ("Classifier Head",  "Dropout(0.5) → Linear(1024→1)"),
            ("Output",           "Sigmoid → threshold 0.5"),
            ("Task",             "Binary: Caries(0) / Healthy(1)"),
        ]
        for lbl, val in rows:
            c1, c2 = st.columns([1,2])
            c1.markdown(f"**{lbl}**")
            c2.markdown(val)
    with col_b:
        st.markdown("**Why DenseNet121?**")
        st.markdown("""
- **Dense connections**: each layer receives feature maps from all preceding layers,
  enabling strong gradient flow and feature reuse.
- **Parameter efficiency**: fewer parameters than ResNet for similar accuracy.
- **ImageNet pretraining**: lower-level texture/edge detectors reused; only
  high-level semantic layers (DenseBlock 4) are fine-tuned on dental X-rays.
- **Grad-CAM friendly**: `norm5` produces clean 7×7 spatial maps that highlight
  clinically relevant tooth regions.
        """)

    # ── DenseBlock explainer ─────────────────────────────────────────────
    with st.expander("📦 What is a Dense Block?"):
        st.markdown("""
A **Dense Block** connects every layer to every other layer in a feed-forward fashion.
If a block has *L* layers, there are *L(L+1)/2* connections.

```
Layer 1  ──────────────────────────► Layer 4
Layer 1 → Layer 2 ─────────────────► Layer 4
Layer 1 → Layer 2 → Layer 3 ───────► Layer 4
```
Each layer concatenates all previous feature maps, so the network learns
both fine-grained and coarse features simultaneously — ideal for subtle
dental caries lesions that appear as small dark spots on X-rays.
        """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — REPORT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Report Analysis":
    st.title("📋 Report Analysis")
    st.markdown("Deep-dive into model performance, confusion matrix breakdown, "
                "and root causes of sub-optimal results.")
    st.markdown("---")

    # Load data
    m = load_json(METRICS) if os.path.exists(METRICS) else {}
    h = load_json(HISTORY) if os.path.exists(HISTORY) else {}

    # ── Summary scorecard ─────────────────────────────────────────────────
    st.subheader("Overall Test-Set Scorecard")
    cm = load_cm()
    tp_c,fn_c,fp_c,tp_h = int(cm[0,0]),int(cm[0,1]),int(cm[1,0]),int(cm[1,1])
    total = tp_c+fn_c+fp_c+tp_h
    acc   = (tp_c+tp_h)/total
    pc = tp_c/(tp_c+fp_c); rc = tp_c/(tp_c+fn_c); f1c = 2*pc*rc/(pc+rc)
    ph = tp_h/(tp_h+fn_c); rh = tp_h/(tp_h+fp_c); f1h = 2*ph*rh/(ph+rh)

    cols = st.columns(6)
    for col,(lbl,val,color) in zip(cols,[
        ("Accuracy",     f"{acc*100:.1f}%",  "#34d399"),
        ("Caries Prec",  f"{pc*100:.1f}%",   "#fb7185"),
        ("Caries Rec",   f"{rc*100:.1f}%",   "#fb7185"),
        ("Healthy Prec", f"{ph*100:.1f}%",   "#34d399"),
        ("Healthy Rec",  f"{rh*100:.1f}%",   "#fbbf24"),
        ("Test Loss",    str(m.get("test_loss","N/A")), "#818cf8"),
    ]):
        col.markdown(f"""<div class="metric-card">
            <h3>{lbl}</h3><p style="color:{color}">{val}</p></div>""",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Confusion matrix heatmap ──────────────────────────────────────────
    st.subheader("Confusion Matrix — Deep Dive")
    ca, cb = st.columns([1,1])
    with ca:
        heatmap = go.Figure(go.Heatmap(
            z=cm, x=["Pred: Caries","Pred: Healthy"],
            y=["True: Caries","True: Healthy"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}", textfont=dict(size=22, color="white"),
            colorscale=[[0,"#0b1120"],[0.5,"#1e3a5f"],[1,"#818cf8"]],
            showscale=False))
        heatmap.update_layout(template="plotly_dark", height=320,
            margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(heatmap, use_container_width=True)
    with cb:
        st.markdown("#### What the numbers mean")
        st.markdown(f"""
| Cell | Count | Interpretation |
|------|-------|----------------|
| TP Caries | **{tp_c}** | Correctly detected caries |
| FN Caries | **{fn_c}** | Caries missed (predicted healthy) |
| FP Caries | **{fp_c}** | Healthy mis-labelled as caries |
| TP Healthy | **{tp_h}** | Correctly identified healthy |

> **Key insight:** The model achieves **98% caries recall** (2 caries missed)
> but mis-classifies **{fp_c} healthy** teeth as caries — a direct consequence
> of the 4:1 class imbalance in the training set.
        """)

    st.markdown("---")

    # ── Per-class bar chart ───────────────────────────────────────────────
    st.subheader("Per-Class Metric Comparison")
    bar = go.Figure()
    bar.add_trace(go.Bar(name="Caries", x=["Precision","Recall","F1 Score"],
        y=[pc*100,rc*100,f1c*100], marker_color="#fb7185",
        text=[f"{v:.1f}%" for v in [pc*100,rc*100,f1c*100]],
        textposition="outside"))
    bar.add_trace(go.Bar(name="Healthy", x=["Precision","Recall","F1 Score"],
        y=[ph*100,rh*100,f1h*100], marker_color="#34d399",
        text=[f"{v:.1f}%" for v in [ph*100,rh*100,f1h*100]],
        textposition="outside"))
    bar.update_layout(template="plotly_dark", height=380, barmode="group",
        yaxis=dict(range=[0,120]),
        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("---")

    # ── Root cause analysis ───────────────────────────────────────────────
    st.subheader("Root Causes of Sub-Optimal Healthy-Class Recall")

    issues = [
        {
            "icon": "🔴",
            "title": "1. Severe Class Imbalance  (Primary Cause)",
            "color": "#fb7185",
            "bg":    "#2d0a0a",
            "points": [
                "Training set: **457 Caries** vs **113 Healthy** — a 4:1 ratio.",
                "Although `WeightedRandomSampler` was applied, val/test sets remain imbalanced.",
                "The model is biased toward predicting Caries, leading to **8 healthy** mis-classifications.",
                "Evidence: Caries Recall = **98%** (2 missed), Healthy Recall = **68%** (8 mis-classified as Caries).",
            ],
        },
        {
            "icon": "🟡",
            "title": "2. Very Small Dataset",
            "color": "#fbbf24",
            "bg":    "#2d1f00",
            "points": [
                "Total dataset: **815 images** (570 train / 122 val / 123 test).",
                "DenseNet121's DenseBlock 4 alone contains **~7 million** trainable parameters.",
                "With only 570 training samples, the model risks memorising rather than generalising.",
                "Rule of thumb: need ~10× more samples per parameter group for robust learning.",
            ],
        },
        {
            "icon": "🟠",
            "title": "3. Partial Fine-Tuning (Frozen Early Layers)",
            "color": "#fb923c",
            "bg":    "#2d1200",
            "points": [
                "Only DenseBlock 4 + Norm5 were unfrozen — blocks 1–3 retain ImageNet weights.",
                "Dental X-rays have very different texture statistics from natural RGB images.",
                "Frozen low-level filters may not extract optimal enamel/lesion edge features.",
                "Full fine-tuning (with a lower LR) would likely improve minority-class recall.",
            ],
        },
    ]

    for iss in issues:
        with st.expander(f"{iss['icon']}  {iss['title']}", expanded=True):
            for pt in iss["points"]:
                st.markdown(f"- {pt}")

    st.markdown("---")

    # ── Training dynamics ─────────────────────────────────────────────────
    st.subheader("Training Dynamics Analysis")
    if h:
        ep = list(range(1, len(h["train_loss"])+1))
        gap = [abs(a-b)*100 for a,b in zip(h["train_acc"], h["val_acc"])]
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("Train vs Val Loss Gap", "Train-Val Accuracy Gap (%)"))
        fig.add_trace(go.Scatter(x=ep, y=h["train_loss"], name="Train Loss",
            line=dict(color="#818cf8",width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=ep, y=h["val_loss"], name="Val Loss",
            line=dict(color="#fb923c",width=2)), 1, 1)
        fig.add_trace(go.Bar(x=ep, y=gap, name="Acc Gap",
            marker_color="#fbbf24"), 1, 2)
        fig.update_layout(template="plotly_dark", height=360,
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "**Loss gap is small** — model does not heavily overfit. "
            "The accuracy gap fluctuates due to the small val set (122 samples). "
            "The LR dropped at epochs 10 and 13 via ReduceLROnPlateau."
        )

    # ── PDF download ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full PDF Report")
    if os.path.exists(PDF_PATH):
        with open(PDF_PATH, "rb") as f:
            st.download_button("📥 Download Full PDF Report", f,
                file_name="DenseNet121_Caries_Report.pdf",
                mime="application/pdf")
    else:
        st.warning("PDF not found. Run `generate_report.py` first.")
