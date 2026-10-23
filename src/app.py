import streamlit as st
import torch
import json
import random
import numpy as np
from model import AgriBotNN
from nltk_utils import bag_of_words, tokenize

# --- Page Configuration ---
st.set_page_config(page_title="AgriBot-CAR", page_icon="🌱", layout="centered")
st.title("🌱 AgriBot-CAR")
st.caption("Cordillera Agricultural Soil Knowledge Base")

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
    with open('src/data/intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
        
    data = torch.load("src/data/agribot_data.pth", map_location=device, weights_only=True)
    model = AgriBotNN(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    
    return model, data, intents, device

model, data, intents, device = load_ai_model()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Naimbag a bigat! Hello! Ask me anything about soil."}
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & AI Response ---
if prompt := st.chat_input("Ask about your soil... / Agdamag maipapan iti daga..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sentence = tokenize(prompt)
    X = bag_of_words(sentence, data['all_words'])
    X = torch.tensor(X.reshape(1, X.shape[0]), dtype=torch.float32).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Determine Response based on Language Tab
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if language_tab == "English":
                    bot_response = random.choice(intent['responses_en'])
                else:
                    bot_response = random.choice(intent['responses_il'])
    else:
        if language_tab == "English":
            bot_response = "I didn't quite understand that. Could you rephrase your question?"
        else:
            bot_response = "Awan ti naan-anay a pannakaawatko. Mabalin kadi a padasem manen?"

    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})