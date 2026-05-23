import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# 1. Download 'punkt' - a small tool that helps Python find sentence boundaries
nltk.download('punkt')

# 2. Setup the "Paraphraser" - This is our AI engine that rewrites text
@st.cache_resource  # This ensures the AI model only loads once so the app is fast
def load_model():
    # We use a T5 model, which is great at rewriting while keeping the same meaning
    return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

paraphraser = load_model()

def humanize_text(text):
    # Split the long block of text into individual sentences
    sentences = sent_tokenize(text)
    humanized_sentences = []
    
    for i, sentence in enumerate(sentences):
        # BURSTINESS LOGIC: We only rewrite every 2nd sentence.
        # This creates a "human rhythm" of one AI-perfect sentence and one rewritten one.
        if i % 2 == 0:
            result = paraphraser(f"paraphrase: {sentence}", max_length=128)
            humanized_sentences.append(result[0]['generated_text'])
        else:
            humanized_sentences.append(sentence)
            
    return " ".join(humanized_sentences)

# 3. STREAMLIT UI - This creates the website look
st.title("🛡️ My Personal AI Humanizer")
st.write("Paste text below to break robotic patterns.")

input_text = st.text_area("Original AI Text", height=200)

if st.button("Humanize Now"):
    if input_text:
        with st.spinner('Mixing in some human chaos...'):
            output = humanize_text(input_text)
            st.subheader("Humanized Version:")
            st.write(output)
    else:
        st.error("Please paste some text first!")
