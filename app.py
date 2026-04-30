import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# --- 1. ESSENTIAL CLOUD SETUP ---
# This fixes the LookupError by downloading required data on the server
nltk.download('vader_lexicon')
nltk.download('punkt')

# --- 2. PAGE CONFIGuration ---
st.set_page_config(page_title="MBA NLP: CSI Calculator", page_icon="📈", layout="wide")

st.title("📈 Customer Satisfaction Index (CSI) Dashboard")
st.markdown("""
*Built for MBA Business Analytics Project*  
This application automates the calculation of a **Customer Satisfaction Index** using Natural Language Processing.
""")

# --- 3. SIDEBAR / INPUT ---
st.sidebar.header("Input Data")
user_input = st.sidebar.text_area(
    "Paste reviews here (One review per line):", 
    height=300,
    placeholder="Example: The service was excellent!\nI am very unhappy with the delay."
)

# --- 4. PROCESSING LOGIC ---
if st.sidebar.button("Generate CSI Analysis"):
    if user_input.strip():
        # Convert input to list
        reviews = [r.strip() for r in user_input.split('\n') if r.strip()]
        
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()
        
        # Calculate Scores
        data = []
        for r in reviews:
            score = analyzer.polarity_scores(r)['compound']
            # CSI Formula: Mapping -1 to +1 scale into 0 to 100 scale
            csi_score = ((score + 1) / 2) * 100
            
            # Labeling for analysis
            if csi_score >= 70: status = "Positive"
            elif csi_score <= 40: status = "Negative"
            else: status = "Neutral"
            
            data.append({"Review": r, "Sentiment_Score": score, "CSI": csi_score, "Status": status})
        
        df = pd.DataFrame(data)
        
        # --- 5. DASHBOARD LAYOUT ---
        col1, col2 = st.columns(2)
        
        with col1:
            avg_csi = df['CSI'].mean()
            st.metric(label="Overall Customer Satisfaction Index", value=f"{avg_csi:.2f}/100")
            
            # CSI Distribution Plot
            fig, ax = plt.subplots()
            sns.histplot(df['CSI'], bins=10, kde=True, color='skyblue', ax=ax)
            plt.title("Distribution of CSI Scores")
            st.pyplot(fig)

        with col2:
            st.write("### Sentiment Breakdown")
            status_counts = df['Status'].value_counts()
            st.bar_chart(status_counts)
            
            # Word Cloud
            all_text = " ".join(df['Review'])
            wordcloud = WordCloud(background_color='white', width=800, height=400).generate(all_text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        # Show raw data table
        st.write("### Detailed Analysis Table")
        st.dataframe(df)
        
    else:
        st.sidebar.error("Please enter at least one review to analyze.")

else:
    st.info("👈 Paste your customer reviews in the sidebar and click 'Generate CSI Analysis' to start.")
