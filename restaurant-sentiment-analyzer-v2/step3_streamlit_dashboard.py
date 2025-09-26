#!/usr/bin/env python3
"""
Minimal Restaurant Review Sentiment Dashboard
No external visualization dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core imports only
import re
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Restaurant Intelligence Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        font-weight: bold;
        padding: 1rem 0;
        border-bottom: 4px solid #2E8B57;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Simple sentiment analyzer
class BasicSentimentAnalyzer:
    def __init__(self):
        self.positive_words = [
            'amazing', 'excellent', 'great', 'love', 'perfect', 'wonderful', 
            'outstanding', 'fantastic', 'incredible', 'best', 'awesome', 'delicious'
        ]
        self.negative_words = [
            'terrible', 'awful', 'worst', 'hate', 'horrible', 'disgusting', 
            'bad', 'poor', 'disappointing', 'never', 'rude', 'slow', 'cold'
        ]
    
    def analyze(self, text):
        if not text:
            return {'score': 0, 'confidence': 0, 'sentiment': 'neutral'}
        
        text_lower = str(text).lower()
        words = text_lower.split()
        
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'confidence': 0, 'sentiment': 'neutral'}
        
        if pos_count + neg_count == 0:
            score = 0
            confidence = 0
        else:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            confidence = (pos_count + neg_count) / total_words
        
        if score > 0.1:
            sentiment = 'positive'
        elif score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'score': score,
            'confidence': min(confidence * 2, 1.0),
            'sentiment': sentiment,
            'positive_words_found': pos_count,
            'negative_words_found': neg_count
        }

# Sample data
@st.cache_data
def load_sample_data():
    reviews = [
        ("Amazing Italian restaurant! The pasta was perfectly cooked and the service was outstanding.", 5, "Tony's Corner Pizza", "Italian"),
        ("Decent food but nothing special. The chicken was okay but a bit dry.", 3, "Corner Cafe", "American"),
        ("Outstanding seafood restaurant! The lobster was incredibly fresh and perfectly prepared.", 5, "Ocean's Bounty", "Seafood"),
        ("Terrible experience from start to finish. We were seated 30 minutes late despite having a reservation.", 1, "Downtown Diner", "American"),
        ("Love this place for brunch! The pancakes are fluffy and the eggs benedict is incredible.", 4, "Sunny Side Cafe", "American"),
        ("Authentic Thai flavors that remind me of my trip to Bangkok! The pad thai is perfectly balanced.", 5, "Thai Garden", "Thai"),
        ("Food was good but the service needs improvement. Our server forgot our drink order.", 3, "Hillside Grill", "American"),
        ("Absolutely love the ambiance here - perfect for a romantic dinner.", 4, "Prime Steakhouse", "Steakhouse"),
        ("Disappointing visit. The sushi didn't taste fresh and the rice was too warm.", 2, "Sakura Sushi", "Japanese"),
        ("Great neighborhood spot! The burgers are juicy and the fries are crispy.", 4, "Local Burger Joint", "American"),
        ("Incredible Mexican food! The tacos are authentic and flavorful.", 5, "Casa Miguel", "Mexican"),
        ("Food quality has declined recently. Used to be one of my favorite spots.", 2, "Blue Moon Cafe", "American"),
        ("Perfect date night spot! Quiet enough to actually talk. Wine selection could be better.", 4, "Bella Vista", "Italian"),
        ("Way too loud! Couldn't hear my friend across the table. Food was good though.", 3, "The Rusty Anchor", "Seafood"),
        ("My mom recommended this place and she was right! Reminds me of childhood comfort food.", 4, "Maria's Kitchen", "Mexican")
    ]
    
    df = pd.DataFrame(reviews, columns=['review_text', 'rating', 'restaurant_name', 'cuisine_type'])
    
    # Add random dates
    dates = []
    for i in range(len(df)):
        days_ago = np.random.randint(1, 365)
        date = datetime.now() - timedelta(days=days_ago)
        dates.append(date.strftime('%Y-%m-%d'))
    
    df['review_date'] = dates
    return df

def main():
    apply_custom_css()
    
    # Header
    st.markdown('''
    <div class="main-header">
        ��️ Restaurant Intelligence Dashboard
    </div>
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <em>Advanced Sentiment Analysis & Business Intelligence Platform</em><br>
        <small>Built with ❤️ by Dev2943 | Version 2.0</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data and analyzer
    df = load_sample_data()
    analyzer = BasicSentimentAnalyzer()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Dashboard Controls")
    page = st.sidebar.selectbox(
        "Select Analysis Page",
        ["🏠 Overview", "🔍 Single Review Analysis", "📊 Restaurant Comparison", "🎯 Business Intelligence"]
    )
    
    if page == "🏠 Overview":
        st.markdown("## 📈 Dataset Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        
        with col2:
            st.metric("Restaurants", df['restaurant_name'].nunique())
            
        with col3:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f}⭐")
            
        with col4:
            st.metric("Cuisines", df['cuisine_type'].nunique())
        
        # Charts using Streamlit's built-in charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
        
        with col2:
            st.markdown("### Cuisine Types")
            cuisine_counts = df['cuisine_type'].value_counts()
            st.bar_chart(cuisine_counts)
        
        # Top restaurants
        st.markdown("### 🏆 Top Rated Restaurants")
        top_restaurants = df.groupby('restaurant_name').agg({
            'rating': 'mean',
            'review_text': 'count'
        }).round(2).sort_values('rating', ascending=False).head(10)
        
        top_restaurants.columns = ['Average Rating', 'Review Count']
        st.dataframe(top_restaurants, use_container_width=True)
    
    elif page == "🔍 Single Review Analysis":
        st.markdown("## 🔍 Individual Review Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            review_text = st.text_area(
                "📝 Enter a restaurant review to analyze:",
                height=150,
                placeholder="Type or paste a restaurant review here...\n\nExample: 'Amazing food and great service! The pasta was perfectly cooked and our server was very attentive. Will definitely be back!'"
            )
            
            analyze_button = st.button("🚀 Analyze Review", type="primary")
        
        with col2:
            st.markdown("### 💡 Analysis Features")
            st.markdown("✅ Real-time sentiment scoring")
            st.markdown("✅ Confidence assessment")  
            st.markdown("✅ Keyword detection")
            st.markdown("✅ Business insights")
            
            st.markdown("### 📊 Sentiment Scale")
            st.markdown("🟢 **Positive**: 0.1 to 1.0")
            st.markdown("🟡 **Neutral**: -0.1 to 0.1") 
            st.markdown("🔴 **Negative**: -1.0 to -0.1")
        
        if analyze_button and review_text.strip():
            with st.spinner("🤖 Analyzing review..."):
                results = analyzer.analyze(review_text)
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_emoji = "😊" if results['sentiment'] == 'positive' else "😞" if results['sentiment'] == 'negative' else "😐"
                st.metric("Sentiment", f"{sentiment_emoji} {results['sentiment'].title()}")
            
            with col2:
                st.metric("Score", f"{results['score']:.3f}")
            
            with col3:
                st.metric("Confidence", f"{results['confidence']:.3f}")
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 Keyword Analysis")
                st.markdown(f"**Positive keywords found:** {results['positive_words_found']}")
                st.markdown(f"**Negative keywords found:** {results['negative_words_found']}")
                
                if results['sentiment'] == 'positive':
                    st.success("✅ This review expresses satisfaction with the restaurant experience.")
                elif results['sentiment'] == 'negative':
                    st.error("❌ This review expresses dissatisfaction with the restaurant experience.")
                else:
                    st.info("ℹ️ This review expresses neutral feelings about the restaurant.")
            
            with col2:
                st.markdown("### 💼 Business Insights")
                
                if results['sentiment'] == 'positive' and results['confidence'] > 0.3:
                    st.markdown("🎯 **High-value positive feedback**")
                    st.markdown("• Consider featuring in marketing")
                    st.markdown("• Ask customer for detailed testimonial")
                elif results['sentiment'] == 'negative' and results['confidence'] > 0.3:
                    st.markdown("🚨 **Requires immediate attention**")
                    st.markdown("• Follow up with customer service")
                    st.markdown("• Investigate specific issues mentioned")
                else:
                    st.markdown("📝 **Standard feedback**")
                    st.markdown("• Monitor for patterns")
                    st.markdown("• Consider requesting more details")
    
    elif page == "📊 Restaurant Comparison":
        st.markdown("## 📊 Restaurant Performance Comparison")
        
        selected_restaurants = st.multiselect(
            "🏪 Select restaurants to compare:",
            options=df['restaurant_name'].unique(),
            default=list(df['restaurant_name'].unique())[:5]
        )
        
        if selected_restaurants:
            filtered_df = df[df['restaurant_name'].isin(selected_restaurants)]
            
            # Performance metrics
            performance = filtered_df.groupby('restaurant_name').agg({
                'rating': ['mean', 'count', 'std']
            }).round(2)
            
            performance.columns = ['Avg Rating', 'Review Count', 'Rating Std Dev']
            performance = performance.sort_values('Avg Rating', ascending=False)
            
            st.markdown("### 📈 Performance Metrics")
            st.dataframe(performance, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Average Ratings")
                avg_ratings = filtered_df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
                st.bar_chart(avg_ratings)
            
            with col2:
                st.markdown("### Review Volume")
                review_counts = filtered_df['restaurant_name'].value_counts()
                st.bar_chart(review_counts)
            
            # Sample reviews
            st.markdown("### 📝 Sample Reviews")
            for restaurant in selected_restaurants[:3]:
                restaurant_reviews = filtered_df[filtered_df['restaurant_name'] == restaurant]
                if not restaurant_reviews.empty:
                    sample_review = restaurant_reviews.iloc[0]
                    
                    with st.expander(f"Sample review for {restaurant}"):
                        st.markdown(f"**Rating:** {'⭐' * sample_review['rating']}")
                        st.markdown(f"**Review:** {sample_review['review_text']}")
                        st.markdown(f"**Cuisine:** {sample_review['cuisine_type']}")
    
    elif page == "🎯 Business Intelligence":
        st.markdown("## 🎯 Business Intelligence Dashboard")
        
        if st.button("🚀 Generate Analysis Report", type="primary"):
            with st.spinner("🤖 Generating insights..."):
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Reviews", f"{len(df):,}")
                
                with col2:
                    avg_rating = df['rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f}⭐")
                
                with col3:
                    st.metric("Restaurants", df['restaurant_name'].nunique())
                
                with col4:
                    st.metric("Cuisines", df['cuisine_type'].nunique())
                
                st.markdown("---")
                
                # Performance analysis
                restaurant_performance = df.groupby('restaurant_name').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                restaurant_performance.columns = ['avg_rating', 'review_count']
                restaurant_performance = restaurant_performance.sort_values('avg_rating', ascending=False)
                
                # Top performer
                if len(restaurant_performance) > 0:
                    top_performer = restaurant_performance.index[0]
                    top_rating = restaurant_performance.iloc[0]['avg_rating']
                    
                    st.success(f"""
                    🌟 **Top Performer: {top_performer}**  
                    Average Rating: {top_rating} stars  
                    **Recommendation:** This restaurant is excelling! Consider using their practices as a benchmark.
                    """)
                
                # Needs attention
                worst_performers = restaurant_performance[restaurant_performance['avg_rating'] < 3.0]
                if not worst_performers.empty:
                    worst_performer = worst_performers.index[0]
                    worst_rating = worst_performers.iloc[0]['avg_rating']
                    
                    st.error(f"""
                    🚨 **Needs Attention: {worst_performer}**  
                    Average Rating: {worst_rating} stars  
                    **Recommendation:** This location requires immediate quality improvement focus.
                    """)
                
                # Cuisine analysis
                cuisine_performance = df.groupby('cuisine_type')['rating'].mean().sort_values(ascending=False)
                
                st.info(f"""
                🍽️ **Cuisine Performance Analysis**  
                Best Performing: {cuisine_performance.index[0]} ({cuisine_performance.iloc[0]:.2f} avg)  
                **Recommendation:** Consider expanding {cuisine_performance.index[0]} offerings or menu items.
                """)
                
                # Detailed performance table
                st.markdown("### 📊 Detailed Performance Analysis")
                st.dataframe(restaurant_performance.sort_values('avg_rating', ascending=False), use_container_width=True)
                
                # Export data
                st.markdown("### 💾 Export Data")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Dataset (CSV)",
                    data=csv,
                    file_name=f"restaurant_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
