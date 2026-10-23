import tkinter as tk
from tkinter import font
import json
import random
import torch
from model import AgriBotNN
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('src/data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "src/data/agribot_data.pth"
data = torch.load(FILE, map_location=device, weights_only=True)

model = AgriBotNN(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

bot_name = "AgriBot-CAR"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, data['all_words'])
    X = torch.tensor(X.reshape(1, X.shape[0]), dtype=torch.float32).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = data['tags'][predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "Awan ti naan-anay a pannakaawatko iti dinamagmo. (I cannot understand clearly, could you rephrase?)"

class ChatApplication:
    def __init__(self):
        self.window = tk.Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("AgriBot-CAR")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg="#1E1E1E")
        
        # Chat History Text Widget
        self.text_widget = tk.Text(self.window, width=20, height=2, bg="#1E1E1E", fg="#FFFFFF",
                                font="Helvetica 12", padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # Bottom Label (Input Area Background)
        bottom_label = tk.Label(self.window, bg="#2C2C2C", height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        # Input Field
        self.msg_entry = tk.Entry(bottom_label, bg="#3C3C3C", fg="#FFFFFF", font="Helvetica 12")
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # Send Button
        send_button = tk.Button(bottom_label, text="Send", font="Helvetica 12 bold", 
                                width=20, bg="#4CAF50", fg="#FFFFFF", command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        
        # Initial Greeting
        self._insert_message("Naimbag a bigat! Ania ti maitulongko maipapan iti dagam? (Hello! How can I help with your soil?)", bot_name)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, tk.END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg1)
        self.text_widget.configure(state=tk.DISABLED)
        
        if sender == "You":
            response = get_response(msg)
            self._insert_message(response, bot_name)
            
        self.text_widget.see(tk.END)

if __name__ == "__main__":
    app = ChatApplication()
    app.run()