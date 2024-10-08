import streamlit as st
import torch
from req import GPTModel, generate_text_simple
import tiktoken
from torch.quantization import quantize_dynamic
import gdown
import os

model_url = "https://drive.google.com/uc?id=1pTPPleG3Q804ZQWkKKo_hTArrxFMYl_7"
model_path = "model.pth"

#Downloading the PyTorch Model
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

#caching the model to optimize the performance
@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("gpt2")


@st.cache_resource
def load_model():
    # Define model configuration
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

    # Load the model
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    # Apply dynamic quantization to optimize the model
    model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return model_quantized, device

# Cached functions for tokenizer and model
tokenizer = load_tokenizer()
model_quantized, device = load_model()

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
#number of tokes to be generated input
token = st.number_input("Enter number of words to be generated", min_value=5, max_value=200, step=1)


# Generate button
if st.button("Generate"):
    if context:
        token_ids = generate_text_simple(
            model=model_quantized,
            idx=text_to_token_ids(context, tokenizer).to(device),
            max_new_tokens=token,
            context_size=1024
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        st.write(generated_text)
        if generated_text:
            st.download_button('Download Text', generated_text, file_name='generated_text.txt')

    else:
        st.write("Please enter a prompt!")

with st.expander("Model Information"):
    st.write("GPT Model: 124M parameters with 12 layers and 12 heads. Quantization applied for performance optimization.")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: fff;
        text-align: center;
    }
    .footer p {
        font-size: 20px
    }
    .footer a{
        text-decoration: none;
        color: #fff;
    }
    </style>
    <div class="footer">
        <p>Made by <a href='https://notishi28.dev'>NotIshi28</a></p>
    </div>
    """,
    unsafe_allow_html=True
)