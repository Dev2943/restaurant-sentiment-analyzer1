#!/usr/bin/env python3
"""
Standalone Restaurant Review Sentiment Dashboard
Self-contained version for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core imports with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

import re
import json
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Restaurant Intelligence Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #1E6B3F;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-left: 10px;
        border-left: 5px solid #4169E1;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #F0F8FF 0%, #E6F3FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #4169E1;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #F0FFF0 0%, #E6FFE6 100%);
        border-left: 6px solid #32CD32;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFFACD 100%);
        border-left: 6px solid #FFD700;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #FFF0F0 0%, #FFE6E6 100%);
        border-left: 6px solid #FF6347;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3CB371 0%, #2E8B57 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    .author-note {
        position: fixed;
        bottom: 10px;
        right: 20px;
        background: rgba(46, 139, 87, 0.9);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        z-index: 999;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Simplified sentiment analyzer
class SimpleSentimentAnalyzer:
    def __init__(self):
        self.positive_words = ['amazing', 'excellent', 'great', 'love', 'perfect', 'wonderful', 'outstanding', 'fantastic', 'incredible', 'best']
        self.negative_words = ['terrible', 'awful', 'worst', 'hate', 'horrible', 'disgusting', 'bad', 'poor', 'disappointing', 'never']
        
        if HAS_NLP:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.has_vader = True
            except:
                self.has_vader = False
        else:
            self.has_vader = False
    
    def analyze_sentiment(self, text):
        if not text:
            return {'compound': 0, 'confidence': 0, 'method': 'none'}
        
        text_lower = str(text).lower()
        
        if self.has_vader:
            try:
                scores = self.sia.polarity_scores(text)
                return {
                    'compound': scores['compound'],
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'confidence': abs(scores['compound']),
                    'method': 'VADER'
                }
            except:
                pass
        
        # Fallback to simple word counting
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            compound = 0
        else:
            compound = (pos_count - neg_count) / (pos_count + neg_count)
        
        return {
            'compound': compound,
            'positive': pos_count / len(text_lower.split()) if text_lower.split() else 0,
            'negative': neg_count / len(text_lower.split()) if text_lower.split() else 0,
            'neutral': 1 - (pos_count + neg_count) / len(text_lower.split()) if text_lower.split() else 1,
            'confidence': abs(compound),
            'method': 'Word Count'
        }

# Sample data generator
@st.cache_data
def create_sample_data():
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
        ("Perfect date night spot! quiet enough to actually talk. Wine selection could be better.", 4, "Bella Vista", "Italian"),
        ("Way too loud! Couldn't hear my friend across the table. Food was good though.", 3, "The Rusty Anchor", "Seafood"),
        ("My mom recommended this place and she was right! reminds me of childhood comfort food.", 4, "Maria's Kitchen", "Mexican")
    ]
    
    df = pd.DataFrame(reviews, columns=['review_text', 'rating', 'restaurant_name', 'cuisine_type'])
    
    # Add dates
    dates = []
    for i in range(len(df)):
        days_ago = np.random.randint(1, 365)
        date = datetime.now() - timedelta(days=days_ago)
        dates.append(date.strftime('%Y-%m-%d'))
    
    df['review_date'] = dates
    df['reviewer_id'] = [f"user_{np.random.randint(1000, 9999)}" for _ in range(len(df))]
    
    return df

# Load data
@st.cache_data
def load_data():
    if os.path.exists('data/restaurant_reviews.csv'):
        try:
            return pd.read_csv('data/restaurant_reviews.csv')
        except:
            pass
    return create_sample_data()

# Visualization functions
def create_rating_distribution(df):
    if not HAS_PLOTLY:
        st.error("Plotly not available for visualizations")
        return None
    
    rating_counts = df['rating'].value_counts().sort_index()
    colors = ['#FF6B6B', '#FF9F40', '#FFCD56', '#4BC0C0', '#36A2EB']
    
    fig = go.Figure(data=[
        go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color=colors,
            text=rating_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Review Rating Distribution",
        xaxis_title="Rating (Stars)",
        yaxis_title="Number of Reviews",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_sentiment_gauge(sentiment_score):
    if not HAS_PLOTLY:
        st.error("Plotly not available for gauge")
        return None
    
    color = "green" if sentiment_score >= 0.1 else "red" if sentiment_score <= -0.1 else "yellow"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 1], 'color': "lightgreen"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    apply_custom_css()
    
    # Header
    st.markdown('''
    <div class="main-header">
        🍽️ Restaurant Intelligence Dashboard
    </div>
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <em>Advanced Sentiment Analysis & Business Intelligence Platform</em><br>
        <small>Built with ❤️ by Dev2943 | Version 2.0</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data and analyzer
    df = load_data()
    analyzer = SimpleSentimentAnalyzer()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Dashboard Controls")
    page = st.sidebar.selectbox(
        "Select Analysis Page",
        ["🏠 Overview", "🔍 Single Review Analysis", "📊 Bulk Analysis", "🎯 Business Intelligence"]
    )
    
    if page == "🏠 Overview":
        st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if HAS_PLOTLY:
                fig_ratings = create_rating_distribution(df)
                if fig_ratings:
                    st.plotly_chart(fig_ratings, use_container_width=True)
            else:
                st.bar_chart(df['rating'].value_counts().sort_index())
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### 📊 Quick Stats")
            st.markdown(f"**📝 Total Reviews:** {len(df):,}")
            st.markdown(f"**🏪 Restaurants:** {df['restaurant_name'].nunique()}")
            st.markdown(f"**🍽️ Cuisines:** {df['cuisine_type'].nunique()}")
            st.markdown(f"**⭐ Avg Rating:** {df['rating'].mean():.2f}")
            
            top_restaurants = df.groupby('restaurant_name').agg({
                'rating': 'mean',
                'review_text': 'count'
            }).round(2).sort_values('rating', ascending=False).head(5)
            
            st.markdown("### 🏆 Top Rated Restaurants")
            for restaurant, data in top_restaurants.iterrows():
                st.markdown(f"**{restaurant}**")
                st.markdown(f"⭐ {data['rating']:.1f} ({data['review_text']} reviews)")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🔍 Single Review Analysis":
        st.markdown('<div class="section-header">🔍 Individual Review Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            review_text = st.text_area(
                "📝 Enter a restaurant review to analyze:",
                height=150,
                placeholder="Type or paste a restaurant review here..."
            )
            
            analyze_button = st.button("🚀 Analyze Review", type="primary")
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### 💡 Tips for Best Results")
            st.markdown("- Include specific details about food, service, atmosphere")
            st.markdown("- Mention emotions and opinions clearly")
            st.markdown("- Use natural language (typos are okay!)")
            st.markdown("- Longer reviews provide more insights")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button and review_text.strip():
            with st.spinner("🤖 Analyzing review..."):
                results = analyzer.analyze_sentiment(review_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if HAS_PLOTLY:
                    fig_gauge = create_sentiment_gauge(results['compound'])
                    if fig_gauge:
                        st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.metric("Sentiment Score", f"{results['compound']:.3f}")
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Sentiment Breakdown")
                st.markdown(f"**Overall Score:** {results['compound']:.3f}")
                st.markdown(f"**Method Used:** {results['method']}")
                st.markdown(f"**Confidence:** {results['confidence']:.3f}")
                if 'positive' in results:
                    st.markdown(f"**Positive:** {results['positive']:.3f}")
                    st.markdown(f"**Negative:** {results['negative']:.3f}")
                    st.markdown(f"**Neutral:** {results['neutral']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🔍 Analysis Summary")
                
                if results['compound'] > 0.1:
                    st.markdown("😊 **Overall Sentiment: POSITIVE**")
                    st.markdown("This review expresses satisfaction with the restaurant experience.")
                elif results['compound'] < -0.1:
                    st.markdown("😞 **Overall Sentiment: NEGATIVE**")
                    st.markdown("This review expresses dissatisfaction with the restaurant experience.")
                else:
                    st.markdown("😐 **Overall Sentiment: NEUTRAL**")
                    st.markdown("This review expresses mixed or neutral feelings about the restaurant.")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📊 Bulk Analysis":
        st.markdown('<div class="section-header">📊 Bulk Analysis</div>', unsafe_allow_html=True)
        
        selected_restaurants = st.multiselect(
            "🏪 Select restaurants to analyze:",
            options=df['restaurant_name'].unique(),
            default=df['restaurant_name'].unique()[:5]
        )
        
        if selected_restaurants:
            filtered_df = df[df['restaurant_name'].isin(selected_restaurants)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                restaurant_ratings = filtered_df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
                if HAS_PLOTLY:
                    fig_comparison = px.bar(
                        x=restaurant_ratings.index,
                        y=restaurant_ratings.values,
                        title="Average Rating by Restaurant"
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                else:
                    st.bar_chart(restaurant_ratings)
            
            with col2:
                review_counts = filtered_df['restaurant_name'].value_counts()
                if HAS_PLOTLY:
                    fig_counts = px.pie(
                        values=review_counts.values,
                        names=review_counts.index,
                        title="Review Distribution"
                    )
                    st.plotly_chart(fig_counts, use_container_width=True)
                else:
                    st.write("Review counts:")
                    st.write(review_counts)
    
    elif page == "🎯 Business Intelligence":
        st.markdown('<div class="section-header">🎯 Business Intelligence Dashboard</div>', unsafe_make_html=True)
        
        if st.button("🚀 Generate Analysis Report", type="primary"):
            with st.spinner("🤖 Generating insights..."):
                
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
                
                st.markdown('<div class="section-header">💡 AI-Generated Recommendations</div>', unsafe_allow_html=True)
                
                # Top performers
                top_restaurants = df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
                worst_restaurants = df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=True)
                
                if len(top_restaurants) > 0:
                    st.markdown(f'''
                    <div class="insight-box success-box">
                        <h4>🌟 Top Performer: {top_restaurants.index[0]}</h4>
                        <p><strong>Average Rating:</strong> {top_restaurants.iloc[0]:.2f} stars</p>
                        <p><strong>Recommendation:</strong> This restaurant is excelling! Consider using their practices as a benchmark for other locations.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                if len(worst_restaurants) > 0 and worst_restaurants.iloc[0] < 3.0:
                    st.markdown(f'''
                    <div class="insight-box danger-box">
                        <h4>🚨 Needs Attention: {worst_restaurants.index[0]}</h4>
                        <p><strong>Average Rating:</strong> {worst_restaurants.iloc[0]:.2f} stars</p>
                        <p><strong>Recommendation:</strong> This restaurant requires immediate attention. Focus on service training and quality control.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Cuisine analysis
                cuisine_performance = df.groupby('cuisine_type')['rating'].mean().sort_values(ascending=False)
                st.markdown(f'''
                <div class="insight-box">
                    <h4>🍽️ Cuisine Performance Analysis</h4>
                    <p><strong>Best Performing Cuisine:</strong> {cuisine_performance.index[0]} ({cuisine_performance.iloc[0]:.2f} avg rating)</p>
                    <p><strong>Growth Opportunity:</strong> Consider expanding {cuisine_performance.index[0]} offerings</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown('''
    <div class="author-note">
        Built by Dev2943 👨‍💻
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    if HAS_NLP:
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
    main()