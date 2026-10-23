import streamlit as st
import torch
import json
import random
import csv
import numpy as np
from pathlib import Path
from model import AgriBotNN
from nltk_utils import bag_of_words, tokenize

# --- Pathing ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- Page Configuration ---
st.set_page_config(page_title="AgriBot-CAR", page_icon="🌱", layout="centered")

# --- UI Setup ---
st.title("🌱 AgriBot-CAR")
st.caption("Cordillera Agricultural Soil Knowledge Base")

# --- Sidebar: Soil Scan Context ---
with st.sidebar:
    st.header("📊 Input Soil Scan Results")
    st.write("Link your thesis results to personalize the chat.")
    n_level = st.selectbox("Nitrogen (N)", ["Unknown", "Low", "Medium", "High"])
    p_level = st.selectbox("Phosphorus (P)", ["Unknown", "Low", "Medium", "High"])
    k_level = st.selectbox("Potassium (K)", ["Unknown", "Low", "Medium", "High"])
    
    if st.button("Apply Context"):
        st.session_state.scan_context = {"N": n_level, "P": p_level, "K": k_level}
        st.success("Scan context updated!")
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Language Switch ---
language_tab = st.radio(
    "Select Language / Piliem ti Pagsasaom:",
    ["English", "Ilokano"],
    horizontal=True
)

# --- Load Model ---
@st.cache_resource
def load_ai_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(DATA_DIR / 'intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
        
    data = torch.load(DATA_DIR / "agribot_data.pth", map_location=device, weights_only=True)
    model = AgriBotNN(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    
    intent_by_tag = {i["tag"]: i for i in intents["intents"]}
    return model, data, intent_by_tag, device

model, data, intent_by_tag, device = load_ai_model()

# --- Helpers ---
def log_unanswered(text):
    log_path = DATA_DIR / "unanswered.csv"
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([text.replace("\n", " ")])

# --- Core Inference Function ---
def get_bot_response(user_text, lang):
    CONF_MIN, MARGIN_MIN = 0.75, 0.25
    
    X = bag_of_words(tokenize(user_text), data["all_words"])
    
    # 1. Out-of-vocabulary check
    if X.sum() == 0:
        log_unanswered(user_text)
        fallback = "I didn't quite catch that. Are you asking about **soil color**, **how to scan**, or **AI accuracy**?" if lang == "English" else "Awan ti naan-anay a pannakaawatko. Agdadamag ka kadi maipapan iti **maris ti daga**, **panag-scan**, wenno **kina-eksakto ti AI**?"
        return fallback, None

    x = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Inference without gradients
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        
    # 3. Top-2 margin check
    (p1, p2), (i1, i2) = torch.topk(probs, 2)
    
    if p1 < CONF_MIN or (p1 - p2) < MARGIN_MIN:
        log_unanswered(user_text)
        tag1, tag2 = data["tags"][i1.item()], data["tags"][i2.item()]
        fallback = f"I didn't catch that. Did you mean **{tag1.replace('_', ' ')}** or **{tag2.replace('_', ' ')}**?" if lang == "English" else f"Diak maawatan. Kayatmo kadi sawen ket **{tag1.replace('_', ' ')}** wenno **{tag2.replace('_', ' ')}**?"
        return fallback, None

    # 4. Dictionary Lookup
    tag = data["tags"][i1.item()]
    intent = intent_by_tag[tag]
    key = "responses_en" if lang == "English" else "responses_il"
    base_response = random.choice(intent[key])
    
    # 5. Context Injection (Thesis Connection)
    if tag == "fertilizer_guidance" and "scan_context" in st.session_state:
        ctx = st.session_state.scan_context
        if lang == "English":
            base_response += f"\n\n**Note on your scan:** Since your Nitrogen is {ctx['N']}, Phosphorus is {ctx['P']}, and Potassium is {ctx['K']}, focus your fertilizer ratio accordingly."
        else:
            base_response += f"\n\n**Palagip iti scan-mo:** Gapu ta ti Nitrogen mo ket {ctx['N']}, Phosphorus ket {ctx['P']}, ken Potassium ket {ctx['K']}, ibasarmo ditoy ti kaadu ti abonem."
            
    return base_response, tag

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    greeting = "Hello! Ask me anything about soil, NPK, or our AI stack." if language_tab == "English" else "Naimbag a bigat! Agdamag maipapan iti daga, NPK, wenno ti AI stack."
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# --- Suggested Prompts UI ---
st.write("### Quick Questions")
col1, col2, col3 = st.columns(3)
suggested_prompt = None

if language_tab == "English":
    if col1.button("How accurate is the AI?"): suggested_prompt = "How accurate is the AI?"
    if col2.button("Scan on paper or container?"): suggested_prompt = "Should I scan on paper or in a container?"
    if col3.button("What is NPK?"): suggested_prompt = "What is NPK?"
else:
    if col1.button("Mamatian kadi daytoy AI?"): suggested_prompt = "Mamatian kadi daytoy AI?"
    if col2.button("Nasaysayaat kadi ti papel?"): suggested_prompt = "Nasaysayaat kadi ti papel wenno container?"
    if col3.button("Ania ti NPK?"): suggested_prompt = "Ania ti NPK?"

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handling ---
placeholder = "Ask about your soil..." if language_tab == "English" else "Agdamag maipapan iti daga..."
user_input = st.chat_input(placeholder)

if suggested_prompt:
    user_input = suggested_prompt

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response, _ = get_bot_response(user_input, language_tab)
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})