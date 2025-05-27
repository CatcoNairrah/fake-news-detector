import streamlit as st
import torch
import torch.nn as nn
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load tokenizer and parameters
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("params.pkl", "rb") as f:
    params = pickle.load(f)

maxlen = params['maxlen']

# Define model class again
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_tensor, hidden_dim=128):
        super().__init__()
        num_embeddings, embedding_dim = embedding_tensor.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_tensor)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze()

# Load embedding matrix
embedding_vectors = np.load("embedding_vectors.npy")
embedding_tensor = torch.tensor(embedding_vectors, dtype=torch.float)

# Instantiate and load model
model = LSTMClassifier(embedding_tensor)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news headline or article text:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=maxlen)
    input_tensor = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    label = "🟢 Real News" if prediction >= 0.5 else "🔴 Fake News"
    st.subheader(f"Prediction: {label} ")
