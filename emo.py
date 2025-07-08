# Optimized emo.py for Business Analytics
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon
from transformers import pipeline

# Ensure NLTK data is downloaded
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download("opinion_lexicon")

# Cache model loading
@st.cache_resource
def get_bert_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def get_emotion_pipeline():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Sentiment analysis
def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score >= 0.2:
        return "Positive"
    elif score <= -0.2:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_bert(text, pipe):
    result = pipe(text[:512])[0]
    return "Positive" if result['label'] == 'LABEL_1' else "Negative"

def detect_emotion(text, pipe):
    results = pipe(text[:512])[0]
    top = max(results, key=lambda x: x['score'])
    return top['label'], top['score']

def process_text(text, bert_pipe, emotion_pipe):
    vader = analyze_sentiment_vader(text)
    bert = analyze_sentiment_bert(text, bert_pipe)
    emotion, confidence = detect_emotion(text, emotion_pipe)
    return vader, bert, emotion, confidence

def main():
    st.set_page_config("Sentiment & Emotion Analyzer - Business")
    st.title("📊 Business Feedback Analyzer")

    st.markdown("""
    Upload a CSV file containing customer feedback (column name must include 'text').
    The app will analyze sentiment and emotion for each entry.
    """)

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        if not text_columns:
            st.error("No column with 'text' found.")
            return

        text_col = text_columns[0]
        st.success(f"Analyzing column: {text_col}")

        bert_pipe = get_bert_pipeline()
        emotion_pipe = get_emotion_pipeline()

        df['VADER_Sentiment'] = df[text_col].astype(str).apply(analyze_sentiment_vader)
        df['BERT_Sentiment'] = df[text_col].astype(str).apply(lambda x: analyze_sentiment_bert(x, bert_pipe))
        df['Emotion'], df['Confidence'] = zip(*df[text_col].astype(str).apply(lambda x: detect_emotion(x, emotion_pipe)))

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "analyzed_results.csv", "text/csv")

if __name__ == "__main__":
    main()
