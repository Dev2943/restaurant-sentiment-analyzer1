import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Restaurant Dashboard", page_icon="🍽️")

st.title("🍽️ Restaurant Intelligence Dashboard")
st.markdown("**Built by Dev2943 | Professional Sentiment Analysis Platform**")

@st.cache_data
def load_data():
    reviews = [
        ("Amazing food! Great service and atmosphere.", 5, "Tony's Pizza", "Italian"),
        ("Decent food but service was slow.", 3, "Corner Cafe", "American"), 
        ("Terrible experience. Cold food, rude staff.", 1, "Downtown Diner", "American"),
        ("Love this place! Best pasta in town.", 5, "Bella Vista", "Italian"),
        ("Good burgers, fair prices.", 4, "Burger Joint", "American")
    ]
    return pd.DataFrame(reviews, columns=['review_text', 'rating', 'restaurant', 'cuisine'])

df = load_data()
page = st.sidebar.selectbox("Select Page", ["Overview", "Review Analysis"])

if page == "Overview":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    with col3:
        st.metric("Restaurants", df['restaurant'].nunique())
    
    st.bar_chart(df['rating'].value_counts())
    st.dataframe(df.groupby('restaurant')['rating'].mean().sort_values(ascending=False))

else:
    review = st.text_area("Enter a restaurant review:", height=100)
    if st.button("Analyze") and review:
        positive_words = ['amazing', 'great', 'excellent', 'love', 'best', 'perfect']
        negative_words = ['terrible', 'awful', 'bad', 'worst', 'hate', 'horrible']
        
        review_lower = review.lower()
        pos_count = sum(word in review_lower for word in positive_words)
        neg_count = sum(word in review_lower for word in negative_words)
        
        if pos_count > neg_count:
            sentiment = "😊 Positive"
        elif neg_count > pos_count:
            sentiment = "😞 Negative" 
        else:
            sentiment = "😐 Neutral"
            
        st.success(f"**Sentiment: {sentiment}**")
        st.write(f"Positive keywords: {pos_count} | Negative keywords: {neg_count}")
