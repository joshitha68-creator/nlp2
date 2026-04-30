import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Page Config
st.set_page_config(page_title="CSI Calculator", page_icon="📊")

st.title("📊 Customer Satisfaction Index (CSI) Calculator")
st.write("Enter customer reviews below to calculate the real-time Satisfaction Index.")

# Text Input
user_input = st.text_area("Paste Customer Reviews here (one per line):")

if st.button("Calculate CSI"):
    if user_input:
        # Process Input
        reviews = user_input.split('\n')
        analyzer = SentimentIntensityAnalyzer()
        
        scores = []
        for r in reviews:
            if r.strip():
                score = analyzer.polarity_scores(r)['compound']
                # Convert to 0-100 scale
                csi = ((score + 1) / 2) * 100
                scores.append(csi)
        
        # Display Results
        avg_csi = sum(scores) / len(scores)
        st.metric(label="Overall CSI Score", value=f"{avg_csi:.2f}/100")
        
        if avg_csi >= 75:
            st.success("High Satisfaction! Keep doing what you're doing.")
        elif avg_csi >= 50:
            st.warning("Neutral Satisfaction. Room for improvement.")
        else:
            st.error("Low Satisfaction. Immediate action required!")
    else:
        st.info("Please enter some text to analyze.")
