import streamlit as st
import torch
from supplementary import GPTModel, generate_text_simple
import tiktoken
import gdown
import os


model_url = "https://drive.google.com/uc?id=1pTPPleG3Q804ZQWkKKo_hTArrxFMYl_7"
model_path = "model.pth"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load the model
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

# Text to token IDs
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor

# Token IDs to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())

# Streamlit App
st.title("LLM Text Generation")

# Input field
context = st.text_input("Enter your text prompt")

# Generate button
if st.button("Generate"):
    if context:
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(context, tokenizer).to(device),
            max_new_tokens=10,
            context_size=128
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        st.write(generated_text)
    else:
        st.write("Please enter a prompt!")