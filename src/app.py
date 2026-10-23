import streamlit as st
import torch
import json
import random
from model import AgriBotNN
from nltk_utils import bag_of_words, tokenize

# --- Page Configuration ---
st.set_page_config(page_title="AgriBot-CAR", page_icon="🌱", layout="centered")
st.title("🌱 AgriBot-CAR")
st.caption("Cordillera Agricultural Soil Analysis Assistant")

# --- Load Model (Cached to prevent reloading on every chat) ---
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
    st.session_state.messages = [
        {"role": "assistant", "content": "Naimbag a bigat! Ania ti maitulongko maipapan iti dagam? (Hello! How can I help with your soil?)"}
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & AI Response ---
if prompt := st.chat_input("Ask about your soil analysis... (e.g., Apay a nababa ti nitrogen?)"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process via PyTorch Model
    sentence = tokenize(prompt)
    X = bag_of_words(sentence, data['all_words'])
    X = torch.tensor(X.reshape(1, X.shape[0]), dtype=torch.float32).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # 3. Determine Response
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
    else:
        bot_response = "Awan ti naan-anay a pannakaawatko. Mabalin kadi a padasem manen? (I didn't quite understand, could you rephrase?)"

    # 4. Display AI Response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})