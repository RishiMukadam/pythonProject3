import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from fpdf import FPDF
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# NLTK downloads
nltk.download("vader_lexicon")

# Load models once
@st.cache_resource
def get_bert_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def get_emotion_pipeline():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Analyze with VADER
def analyze_vader(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    return "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"

# Analyze with BERT
def analyze_bert(text):
    result = get_bert_pipeline()(text[:512])[0]
    return result['label'], result['score']

# Analyze Emotion
def detect_emotion(text):
    result = get_emotion_pipeline()(text[:512])[0]
    top_result = max(result, key=lambda x: x['score'])
    return top_result['label'], top_result['score']

# CSV download link
def generate_csv(data):
    csv = pd.DataFrame(data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis.csv">Download Results as CSV</a>'

def main():
    st.set_page_config(page_title="Customer Sentiment Analyzer", layout="wide")
    st.title("Customer Sentiment Analyzer")

    # Upload
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    data_rows = []
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            selected_col = st.selectbox("Select column to analyze", df.columns)
            row_limit = st.slider("Number of rows to analyze", 1, min(len(df), 1000), len(df))
            data_rows = df[selected_col].dropna().astype(str).tolist()[:row_limit]

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Analyze button
    if st.button("Analyze") and data_rows:
        st.info(f"Analyzing {len(data_rows)} entries...")
        results = []
        progress = st.progress(0)

        for i, text in enumerate(data_rows):
            try:
                vader = analyze_vader(text)
                bert_label, bert_score = analyze_bert(text)
                emotion, emo_score = detect_emotion(text)

                results.append({
                    "Text": text[:100],
                    "VADER": vader,
                    "BERT Sentiment": bert_label,
                    "BERT Confidence": f"{bert_score:.2%}",
                    "Top Emotion": emotion,
                    "Emotion Confidence": f"{emo_score:.2%}"
                })

            except Exception as e:
                results.append({
                    "Text": text[:100],
                    "Error": str(e)
                })

            progress.progress((i + 1) / len(data_rows))

        # Show results
        result_df = pd.DataFrame(results)
        st.dataframe(result_df)
        st.markdown(generate_csv(results), unsafe_allow_html=True)

    # Clear Button
    if st.button("Clear"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
