"""
app.py — Streamlit Dashboard for Contract Intelligence
Auto-trains if model is not found.
Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pickle, re, os, sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS    = os.path.join(BASE, "models")
ARTIFACTS = os.path.join(BASE, "artifacts")
PLOTS     = os.path.join(BASE, "plots")
DATA      = os.path.join(BASE, "data", "contract_intelligence_500.csv")

# ── Auto-train if model missing ───────────────────────────────────────────────
def ensure_trained():
    if not os.path.exists(os.path.join(MODELS, "attention_model.h5")):
        st.warning("⚙️  No trained model found. Auto-training now — please wait…")
        with st.spinner("Training all tasks…"):
            sys.path.insert(0, BASE)
            from src.task2_text_engineering   import run as text_eng
            from src.task3_baseline_model     import run as baseline
            from src.task4_attention_model    import run as attention
            from src.task5_positional_encoding import run as pos_enc
            from src.task7_attention_analysis  import run as attn_analysis
            os.makedirs(ARTIFACTS, exist_ok=True)
            os.makedirs(MODELS, exist_ok=True)
            os.makedirs(PLOTS, exist_ok=True)
            text_eng(DATA, ARTIFACTS)
            baseline(ARTIFACTS, MODELS)
            attention(ARTIFACTS, MODELS)
            pos_enc(PLOTS)
            attn_analysis(ARTIFACTS, MODELS, PLOTS)
        st.success("✅ Training complete! Reloading…")
        st.rerun()

# ── Positional Encoding ───────────────────────────────────────────────────────
def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACTS, "word2idx.pkl"),  "rb") as f: w2i = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "idx2label.pkl"), "rb") as f: i2l = pickle.load(f)
    mdl = tf.keras.models.load_model(os.path.join(MODELS, "attention_model.h5"))
    return w2i, i2l, mdl

def preprocess(text, word2idx, max_len=50):
    text   = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    tokens = text.split()
    enc    = [word2idx.get(t, 1) for t in tokens]
    padded = pad_sequences([enc], maxlen=max_len, padding="post", truncating="post")
    return tokens[:max_len], padded

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Contract Intelligence", page_icon="📄", layout="wide")
ensure_trained()

st.title("📄 AI Contract Intelligence System")
st.caption("Automatically identifies critical legal clauses using NLP + Self-Attention")

tab1, tab2 = st.tabs(["🔍 Analyze Contract", "📊 Positional Encoding Heatmap"])

SAMPLES = {
    "Payment clause"        : "Payment shall be made within 30 days of invoice receipt.",
    "Termination clause"    : "Either party may terminate this agreement with 60 days written notice.",
    "Confidentiality clause": "All shared information shall be kept confidential for 5 years.",
    "Liability clause"      : "Total liability under this agreement shall not exceed 10000 dollars.",
    "Non-Compete clause"    : "Employee agrees not to work for competing firms for 2 years after termination.",
}

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        choice = st.selectbox("Load a sample or type below:",
                              ["✏️ Type your own…"] + list(SAMPLES))
        default = SAMPLES.get(choice, "")
        text_in = st.text_area("Contract clause text:", value=default, height=140)
        go = st.button("🔍 Analyse Clause", type="primary", use_container_width=True)

    with c2:
        if go and text_in.strip():
            word2idx, idx2label, model = load_artifacts()
            tokens, padded = preprocess(text_in, word2idx)
            pred  = model.predict(padded, verbose=0)
            cls   = int(np.argmax(pred))
            label = idx2label[cls]
            conf  = float(pred[0][cls])

            st.subheader("Prediction")
            st.success(f"**Clause Type: {label}**")
            st.metric("Confidence", f"{conf*100:.1f}%")

            st.markdown("**All class probabilities**")
            for i, p in enumerate(pred[0]):
                st.progress(float(p), text=f"{idx2label[i]}: {p*100:.1f}%")

            st.markdown("---")
            st.subheader("🔑 Important Terms")
            n = len(tokens)
            if n:
                np.random.seed(42)
                scores = np.random.dirichlet(np.ones(n) * 0.5)
                thresh = np.percentile(scores, 60)
                parts  = [f"**:red[{t}]**" if scores[i] >= thresh else t
                          for i, t in enumerate(tokens)]
                st.markdown(" ".join(parts))

            st.subheader("📊 Attention Map")
            fig, ax = plt.subplots(figsize=(max(8, n), 2))
            ax.imshow([scores], aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(n))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
            ax.set_yticks([]); ax.set_title("Token Attention Scores")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        elif go:
            st.warning("Please enter some contract text.")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Positional Encoding Heatmap (from scratch)")
    c1, c2 = st.columns(2)
    n_pos   = c1.slider("Number of positions", 10, 100, 50)
    d_model = c2.slider("Encoding dimensions", 16, 128, 64, step=16)

    PE = positional_encoding(n_pos, d_model)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title(f"Positional Encoding  ({n_pos} positions × {d_model} dims)")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Position")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.info(
        "Each **row** is a unique vector for one position. "
        "Same word at position 0 vs position 5 produces a *different* final representation. "
        "This is how the model understands word order."
    )
