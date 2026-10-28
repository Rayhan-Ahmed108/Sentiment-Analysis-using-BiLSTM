import torch
import streamlit as st
import pandas as pd

from data_utils import load_vocab, tokenize, encode_text
from model import SentimentModel


st.set_page_config(page_title="Attention Sentiment Demo")
st.title("Movie Review Sentiment Analysis")
st.write("Enter a review to see the prediction and attention weights.")

@st.cache_resource
def load_model():
    vocab = load_vocab()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = SentimentModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(
        torch.load(
            "artifacts/model.pt",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    return model, vocab, device


model, vocab, device = load_model()

review = st.text_area(
    "Your movie review",
    "The movie was wonderful and the acting was excellent."
)

if st.button("Analyze review"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        tokens = tokenize(review)[:200]
        ids = encode_text(review, vocab, max_length=200)

        input_ids = torch.tensor(
            [ids], dtype=torch.long, device=device
        )
        lengths = torch.tensor([len(ids)])

        with torch.no_grad():
            logits, weights = model(input_ids, lengths)
            probabilities = torch.softmax(logits, dim=1)[0]

        prediction = int(logits.argmax(dim=1).item())
        sentiment = "Positive" if prediction == 1 else "Negative"

        st.subheader(f"Prediction: {sentiment}")
        st.write(
            f"Confidence (model probability): "
            f"{probabilities[prediction].item():.2%}"
        )

        attention_weights = weights[0].cpu().tolist()

        results = pd.DataFrame({
            "Token": tokens,
            "Attention weight": attention_weights[:len(tokens)]
        })

        st.subheader("Attention weights")
        st.caption(
            "Higher weight means the model gave that token "
            "more weight in its context vector. It is not "
            "guaranteed to explain the prediction."
        )
        st.bar_chart(results.set_index("Token"))
        st.dataframe(results, hide_index=True)
