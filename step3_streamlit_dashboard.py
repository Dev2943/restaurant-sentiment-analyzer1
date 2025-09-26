#!/usr/bin/env python3
"""
Step 3: Professional Restaurant Review Sentiment Dashboard
Enhanced Streamlit app with custom styling and professional features

Author: [Your Name]
Date: September 2025
Version: 2.0 - Major UI/UX improvements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Import our custom analyzer
try:
    from step2_sentiment_analyzer import RestaurantSentimentAnalyzer
except ImportError:
    st.error("❌ Please ensure step2_sentiment_analyzer.py is in the same directory!")
    st.stop()

import base64
import io
import json
from datetime import datetime, timedelta
import time
import os

# Set page config
st.set_page_config(
    page_title="Restaurant Intelligence Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def apply_custom_css():
    """Apply custom CSS to make the app look professional and unique"""
    st.markdown("""
    <style>
    /* Main header styling */
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
    
    /* Subheader styling */
    .section-header {
        font-size: 1.8rem;
        color: #1E6B3F;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-left: 10px;
        border-left: 5px solid #4169E1;
    }
    
    /* Insight boxes */
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
    
    /* Custom buttons */
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #F8F9FA 0%, #E9ECEF 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Personal branding */
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
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the restaurant review data"""
    try:
        df = pd.read_csv('data/restaurant_reviews.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run Step 1 to generate the data first.")
        st.stop()

@st.cache_resource
def initialize_analyzer():
    """Initialize and cache the sentiment analyzer"""
    analyzer = RestaurantSentimentAnalyzer()
    
    # Try to load pre-trained models
    if os.path.exists('models/vectorizer.pkl'):
        analyzer.load_models()
    
    return analyzer

def create_sentiment_gauge(sentiment_score, title="Sentiment Score"):
    """Create a professional gauge chart for sentiment"""
    # Determine color based on sentiment
    if sentiment_score >= 0.1:
        color = "green"
    elif sentiment_score <= -0.1:
        color = "red"
    else:
        color = "yellow"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        delta = {'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_rating_distribution(df):
    """Create an enhanced rating distribution chart"""
    rating_counts = df['rating'].value_counts().sort_index()
    
    # Custom colors for each rating
    colors = ['#FF6B6B', '#FF9F40', '#FFCD56', '#4BC0C0', '#36A2EB']
    
    fig = go.Figure(data=[
        go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color=colors,
            text=rating_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x} Stars</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(count/len(df))*100 for count in rating_counts.values]
        )
    ])
    
    fig.update_layout(
        title="Review Rating Distribution",
        xaxis_title="Rating (Stars)",
        yaxis_title="Number of Reviews",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_sentiment_vs_rating_scatter(df, analyzer):
    """Create scatter plot of sentiment vs rating"""
    # Calculate sentiment scores for sample of reviews (to avoid performance issues)
    sample_size = min(200, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    sentiment_scores = []
    for _, row in df_sample.iterrows():
        sentiment = analyzer.safe_sentiment_analysis(row['review_text'])
        sentiment_scores.append(sentiment['compound'])
    
    df_sample['sentiment_score'] = sentiment_scores
    
    fig = px.scatter(
        df_sample,
        x='rating',
        y='sentiment_score',
        color='cuisine_type',
        hover_data=['restaurant_name'],
        title="Sentiment Score vs Star Rating",
        labels={'rating': 'Star Rating', 'sentiment_score': 'Sentiment Score'}
    )
    
    fig.update_layout(height=400)
    return fig

def analyze_review_text(text, analyzer):
    """Analyze a single review and return comprehensive results"""
    results = {
        'sentiment': analyzer.safe_sentiment_analysis(text),
        'aspects': analyzer.analyze_aspects(text),
        'ml_prediction': None
    }
    
    # Try ML prediction if models are available
    if analyzer.is_trained:
        results['ml_prediction'] = analyzer.predict_ml_sentiment(text)
    
    return results

def display_business_insights(insights):
    """Display business insights in an attractive format"""
    if 'error' in insights:
        st.error(f"❌ Error generating insights: {insights['error']}")
        return
    
    st.markdown('<div class="section-header">📊 Business Intelligence Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reviews",
            value=f"{insights['overview']['total_reviews']:,}",
            delta=None
        )
    
    with col2:
        avg_rating = insights['overview']['avg_rating']
        st.metric(
            label="Average Rating",
            value=f"{avg_rating:.2f}⭐",
            delta=f"{avg_rating - 3.0:.2f}" if avg_rating != 3.0 else None
        )
    
    with col3:
        avg_sentiment = insights['sentiment_trends']['avg_sentiment']
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{avg_sentiment:.3f}" if avg_sentiment != 0 else None
        )
    
    with col4:
        st.metric(
            label="Restaurants",
            value=insights['overview']['unique_restaurants'],
            delta=None
        )
    
    # Recommendations section
    if insights.get('recommendations'):
        st.markdown('<div class="section-header">💡 AI-Generated Recommendations</div>', 
                    unsafe_allow_html=True)
        
        for i, rec in enumerate(insights['recommendations'], 1):
            if rec['type'] == 'urgent':
                box_class = 'danger-box'
                icon = '🚨'
            elif rec['type'] == 'improvement':
                box_class = 'warning-box'
                icon = '⚠️'
            else:
                box_class = 'success-box'
                icon = '💡'
            
            st.markdown(f'''
            <div class="insight-box {box_class}">
                <h4>{icon} Recommendation #{i}: {rec['restaurant']}</h4>
                <p><strong>Issue:</strong> {rec['issue']}</p>
                <p><strong>Action:</strong> {rec['action']}</p>
            </div>
            ''', unsafe_allow_html=True)

def create_wordcloud(text_data, title="Word Cloud"):
    """Create a professional word cloud"""
    try:
        # Combine all text
        text = ' '.join(text_data.astype(str))
        
        # Create word cloud with custom styling
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            normalize_plurals=False
        ).generate(text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    except Exception as e:
        st.error(f"❌ Could not generate word cloud: {str(e)}")
        return None

def main():
    """Main dashboard application"""
    # Apply custom styling
    apply_custom_css()
    
    # Header
    st.markdown('''
    <div class="main-header">
        🍽️ Restaurant Intelligence Dashboard
    </div>
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <em>Advanced Sentiment Analysis & Business Intelligence Platform</em><br>
        <small>Built with ❤️ by [Your Name] | Version 2.0</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize components
    try:
        df = load_data()
        analyzer = initialize_analyzer()
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Dashboard Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Analysis Page",
        ["🏠 Overview", "🔍 Single Review Analysis", "📊 Bulk Analysis", "🎯 Business Intelligence"]
    )
    
    if page == "🏠 Overview":
        st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Rating distribution
            fig_ratings = create_rating_distribution(df)
            st.plotly_chart(fig_ratings, use_container_width=True)
            
            # Sentiment vs Rating scatter
            with st.spinner("🔄 Analyzing sentiment patterns..."):
                fig_scatter = create_sentiment_vs_rating_scatter(df, analyzer)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### 📊 Quick Stats")
            st.markdown(f"**📝 Total Reviews:** {len(df):,}")
            st.markdown(f"**🏪 Restaurants:** {df['restaurant_name'].nunique()}")
            st.markdown(f"**🍽️ Cuisines:** {df['cuisine_type'].nunique()}")
            st.markdown(f"**⭐ Avg Rating:** {df['rating'].mean():.2f}")
            
            # Top restaurants
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
        
        # Input section
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
                results = analyze_review_text(review_text, analyzer)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment gauge
                sentiment_score = results['sentiment']['compound']
                fig_gauge = create_sentiment_gauge(sentiment_score)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Detailed sentiment breakdown
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Sentiment Breakdown")
                st.markdown(f"**Overall Score:** {sentiment_score:.3f}")
                st.markdown(f"**Positive:** {results['sentiment']['positive']:.3f}")
                st.markdown(f"**Negative:** {results['sentiment']['negative']:.3f}")
                st.markdown(f"**Neutral:** {results['sentiment']['neutral']:.3f}")
                st.markdown(f"**Confidence:** {results['sentiment']['confidence']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Aspect analysis
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🔍 Aspect Analysis")
                
                aspects = results['aspects']
                for aspect, data in aspects.items():
                    if data['mentions'] > 0:
                        sentiment_emoji = "😊" if data['sentiment'] > 0.1 else "😞" if data['sentiment'] < -0.1 else "😐"
                        st.markdown(f"**{aspect.title()}** {sentiment_emoji}")
                        st.markdown(f"Sentiment: {data['sentiment']:.3f}")
                        st.markdown(f"Keywords: {', '.join(data['keywords'][:5])}")
                        st.markdown("---")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ML Prediction (if available)
            if results['ml_prediction']:
                st.markdown('<div class="insight-box success-box">', unsafe_allow_html=True)
                st.markdown("### 🤖 Machine Learning Prediction")
                ml_pred = results['ml_prediction']
                st.markdown(f"**Predicted Category:** {ml_pred['prediction'].title()}")
                st.markdown(f"**Confidence:** {ml_pred['confidence']:.3f}")
                
                # Show probability distribution
                prob_df = pd.DataFrame(list(ml_pred['probabilities'].items()), 
                                     columns=['Category', 'Probability'])
                fig_prob = px.bar(prob_df, x='Category', y='Probability', 
                                 title="Prediction Probabilities")
                st.plotly_chart(fig_prob, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📊 Bulk Analysis":
        st.markdown('<div class="section-header">📊 Bulk Analysis</div>', unsafe_allow_html=True)
        
        # Restaurant selector
        selected_restaurants = st.multiselect(
            "🏪 Select restaurants to analyze:",
            options=df['restaurant_name'].unique(),
            default=df['restaurant_name'].unique()[:5]
        )
        
        if selected_restaurants:
            filtered_df = df[df['restaurant_name'].isin(selected_restaurants)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating comparison
                restaurant_ratings = filtered_df.groupby('restaurant_name')['rating'].mean().sort_values(ascending=False)
                fig_comparison = px.bar(
                    x=restaurant_ratings.index,
                    y=restaurant_ratings.values,
                    title="Average Rating by Restaurant"
                )
                fig_comparison.update_layout(xaxis_title="Restaurant", yaxis_title="Average Rating")
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                # Review count
                review_counts = filtered_df['restaurant_name'].value_counts()
                fig_counts = px.pie(
                    values=review_counts.values,
                    names=review_counts.index,
                    title="Review Distribution"
                )
                st.plotly_chart(fig_counts, use_container_width=True)
            
            # Word clouds
            st.markdown('<div class="section-header">☁️ Word Cloud Analysis</div>', unsafe_allow_html=True)
            
            wordcloud_type = st.selectbox(
                "Select word cloud type:",
                ["All Reviews", "Positive Reviews (4-5 stars)", "Negative Reviews (1-2 stars)"]
            )
            
            if wordcloud_type == "All Reviews":
                text_data = filtered_df['review_text']
            elif wordcloud_type == "Positive Reviews (4-5 stars)":
                text_data = filtered_df[filtered_df['rating'] >= 4]['review_text']
            else:
                text_data = filtered_df[filtered_df['rating'] <= 2]['review_text']
            
            if not text_data.empty:
                fig_wordcloud = create_wordcloud(text_data, f"{wordcloud_type} Word Cloud")
                if fig_wordcloud:
                    st.pyplot(fig_wordcloud)
    
    elif page == "🎯 Business Intelligence":
        st.markdown('<div class="section-header">🎯 AI-Powered Business Intelligence</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Full Analysis Report", type="primary"):
            with st.spinner("🤖 AI is analyzing all reviews and generating insights..."):
                insights = analyzer.generate_insights(df)
                display_business_insights(insights)
                
                # Additional charts
                st.markdown('<div class="section-header">📈 Performance Trends</div>', unsafe_allow_html=True)
                
                # Convert date column and create time series
                df['review_date'] = pd.to_datetime(df['review_date'])
                df_monthly = df.groupby(df['review_date'].dt.to_period('M')).agg({
                    'rating': 'mean',
                    'review_text': 'count'
                }).reset_index()
                df_monthly['review_date'] = df_monthly['review_date'].astype(str)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_trend = px.line(
                        df_monthly,
                        x='review_date',
                        y='rating',
                        title="Average Rating Trend Over Time"
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    fig_volume = px.bar(
                        df_monthly,
                        x='review_date',
                        y='review_text',
                        title="Review Volume Over Time"
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                # Export functionality
                st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
                
                export_data = {
                    'analysis_date': datetime.now().isoformat(),
                    'dataset_summary': {
                        'total_reviews': len(df),
                        'restaurants': df['restaurant_name'].nunique(),
                        'avg_rating': df['rating'].mean()
                    },
                    'insights': insights
                }
                
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📊 Download Analysis Report (JSON)",
                    data=export_json,
                    file_name=f"restaurant_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Footer with personal branding
    st.markdown('''
    <div class="author-note">
        Built by [Your Name] 👨‍💻
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()