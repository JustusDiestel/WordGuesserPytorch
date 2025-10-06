from transformers import pipeline
import warnings
import streamlit as st

st.title("Word-Guesser")

warnings.filterwarnings("ignore")
fill = pipeline("fill-mask", model="deepset/gbert-base")

text = st.text_input("Gib ein Satz ein und markiere das gesuchte wort mit [MASK]:")

if st.button("START"):
    result = fill(text)
    for w in result:
        st.write(w["token_str"] + ": " + w["sequence"])