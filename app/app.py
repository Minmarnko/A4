import streamlit as st
import torch
import pickle
import os
import torch.nn as nn
from utils import BERT, SimpleTokenizer, calculate_similarity, predict_nli

# ✅ Set Page Config FIRST (before any other Streamlit command)
st.set_page_config(page_title="Text Analysis", layout="centered")

# Cache the model and tokenizer to avoid reloading
@st.cache_resource
def load_model():
    """Loads the trained BERT model and tokenizer."""
    base_path = os.path.dirname(__file__)
    param_path = os.path.join(base_path, "models", "data.pkl")
    model_path = os.path.join(base_path, "models", "s_model.pt")

    if not os.path.exists(param_path) or not os.path.exists(model_path):
        raise FileNotFoundError("⚠️ Model files not found! Ensure 'data.pkl' and 's_model.pt' exist in 'models/'.")

    with open(param_path, "rb") as f:
        data = pickle.load(f)
    
    tokenizer = SimpleTokenizer(data["word2id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERT().to(device)
    state_dict = torch.load(model_path, map_location=device)

    expected_size = 768  
    if model.classifier.in_features != expected_size:
        print(f"⚠️ Adjusting classifier: Expected {expected_size}, Found {model.classifier.in_features}")
        model.classifier = nn.Linear(expected_size, 2).to(device)

    model.load_state_dict(state_dict, strict=False)

    return model, tokenizer, device

# Load model once and cache it
model, tokenizer, device = load_model()

# Streamlit UI
st.title("📖 Sentence Similarity & Logical Relationship")
st.write("Analyze sentence **similarity** and **Natural Language Inference (NLI)**.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    premise = st.text_area("📝 Premise", "The weather is pleasant today with a cool breeze.", height=150)
with col2:
    hypothesis = st.text_area("📝 Hypothesis", "It's a nice day outside with a gentle wind blowing.", height=150)

st.divider()

if st.button("🔍 Analyze Sentences"):
    if premise and hypothesis:
        with st.spinner("Processing... 🔄"):
            similarity_score = calculate_similarity(model, tokenizer, premise, hypothesis, device)
            nli_label, confidence = predict_nli(model, tokenizer, premise, hypothesis, device)

        st.success("✅ Analysis Completed!")
        st.subheader(f"📌 NLI Relationship: **{nli_label}**")
        st.write(f"📊 Confidence: `{confidence:.2f}`")
        st.subheader(f"📏 Sentence Similarity Score: `{similarity_score:.3f}`")
    else:
        st.warning("⚠️ Please enter both sentences before submitting.")

st.divider()
st.caption("🛠 Built with Streamlit & PyTorch | Model: BERT")
