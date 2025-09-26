#!/usr/bin/env python3
"""
Step 1: Authentic Restaurant Review Data Generator
Run this first to create realistic dataset
"""

import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import json
import os

class AuthenticDataGenerator:
    def __init__(self):
        self.restaurants = [
            # Mix of real-sounding names (not actual businesses)
            "Tony's Corner Pizza", "Blue Moon Cafe", "Riverside Diner", "The Rusty Anchor",
            "Maria's Kitchen", "Downtown Burger Co", "Sakura House", "El Corazón",
            "Green Garden Bistro", "The Clock Tower", "Pasta Plus", "Smoky Joe's BBQ",
            "Bella Vista", "Corner Cafe", "Ocean's Bounty", "Downtown Diner",
            "Sunny Side Cafe", "Thai Garden", "Hillside Grill", "Prime Steakhouse",
            "Local Burger Joint", "Casa Miguel", "Golden Dragon", "The Breakfast Spot"
        ]
        
        self.cuisines = {
            "Tony's Corner Pizza": "Italian", "Blue Moon Cafe": "American", 
            "Riverside Diner": "American", "The Rusty Anchor": "Seafood",
            "Maria's Kitchen": "Mexican", "Downtown Burger Co": "American", 
            "Sakura House": "Japanese", "El Corazón": "Mexican",
            "Green Garden Bistro": "Vegetarian", "The Clock Tower": "American", 
            "Pasta Plus": "Italian", "Smoky Joe's BBQ": "BBQ",
            "Bella Vista": "Italian", "Corner Cafe": "American", 
            "Ocean's Bounty": "Seafood", "Downtown Diner": "American",
            "Sunny Side Cafe": "American", "Thai Garden": "Thai", 
            "Hillside Grill": "American", "Prime Steakhouse": "Steakhouse",
            "Local Burger Joint": "American", "Casa Miguel": "Mexican",
            "Golden Dragon": "Chinese", "The Breakfast Spot": "American"
        }

    def generate_authentic_reviews(self):
        """Generate realistic reviews with human imperfections"""
        
        # Real human-style reviews with typos, emotions, and personal details
        authentic_reviews = [
            # 5-star reviews (enthusiastic)
            "OMG this place is AMAZING!! 😍 Went there for my anniversary and everything was perfect. The steak was cooked exactly how I wanted it and the service was incredible. Sarah (our server) was so attentive. Definitely coming back!",
            "Best thai food I've had since my trip to Bangkok! The pad thai was incredible - not too sweet, perfect spice level. My boyfriend loved the green curry too. Hidden gem for sure 💎",
            "Love love love this place! Been coming here for 3 years now and it never disappoints. Maria (the owner) always remembers my usual order. The atmosphere is so cozy and romantic. Perfect date spot ❤️",
            "Incredible seafood! The lobster was so fresh and perfectly cooked. Yes it's expensive but totally worth it for special occasions. Wine selection is also impressive. Will definitely be back!",
            "Amazing brunch spot!! The pancakes are to die for and the eggs benedict is incredible. Coffee is always perfect. Only downside is it gets super crowded on weekends so make a reservation!",
            
            # 4-star reviews (positive but honest)
            "Really good food and nice atmosphere. The steaks are cooked perfectly and the sides are creative. Service is professional but the prices are pretty high. Good for special occasions though.",
            "Great neighborhood spot! Burgers are juicy and fries are crispy. Staff always remembers the regulars which is nice. Fair prices and good portions. Perfect for casual dinner with friends.",
            "Solid Italian place. Pasta was good, not amazing but definitely satisfying. Service was friendly and efficient. Would come back but not necessarily my first choice in the area.",
            "Love the ambiance here - very romantic setting. Food was really good, especially the chicken parmesan. Only complaint is that it can get quite noisy when busy.",
            "Good Mexican food! Tacos are authentic and flavorful. Guacamole made fresh at the table is a nice touch. Margaritas could be stronger but overall good experience.",
            
            # 3-star reviews (mixed feelings)
            "Decent food but nothing special. The chicken was okay but a bit dry. Service was adequate but not particularly friendly. Prices are reasonable for portion sizes. It's fine for a quick meal.",
            "Food was good but service needs improvement. Our server forgot our drink order and we had to ask for condiments multiple times. Food came out hot though. With better service could be 4 stars.",
            "Went here twice now... first time was great, second time not so much. Maybe just an off day? Food quality seems inconsistent. The atmosphere is nice though.",
            "Meh... expected better based on the reviews. Food was okay I guess but nothing stood out. Service was slow and the restaurant was uncomfortably warm. Probably won't go back.",
            "Pretty standard american food. Nothing wrong with it but nothing exciting either. Good option if you have kids since they have a decent kids menu.",
            
            # 2-star reviews (disappointed)
            "Disappointing visit. The sushi didn't taste fresh and the rice was too warm. Service was slow and inattentive. For the price point I expected much better quality.",
            "Food quality has declined recently. This used to be one of my favorite spots but new management has really changed things. Miss the old menu 😢",
            "Reservation system is broken. Waited 45 minutes WITH a reservation. Food was good when we finally got it but seriously need to fix the wait times.",
            "Cash only - heads up! Food is decent but who carries cash anymore?? There's an ATM inside but it's still annoying. Food was mediocre anyway.",
            "Way too loud! Couldn't hear my friend across the table. Food was good but the noise level made it impossible to enjoy. Not going back.",
            
            # 1-star reviews (angry/disappointed)
            "AVOID AT ALL COSTS! Worst meal ever. Food was cold, service was rude, and the tables were dirty. How is this place still open??? Complete waste of money.",
            "Terrible experience from start to finish. Seated 30 minutes late despite reservation. Appetizers were clearly microwaved and main courses were tasteless. Server seemed annoyed when we asked questions.",
            "Food poisoning! Got sick after eating here. The chicken tasted off but I thought I was being paranoid. Spent the whole next day in bed. Never again.",
            "Absolutely awful. Ordered delivery and it took 2 hours to arrive cold. Called to complain and they hung up on me. Worst customer service ever.",
            "Do NOT eat here. Found a hair in my salad and when I told the server they just shrugged. Disgusting. Health department should shut this place down.",
            
            # More varied reviews with personal touches
            "came here after my daughter's graduation. staff was super accommodating for our large group (12 people!). food came out together which was impressive. definitely recommend for groups.",
            "my mom recommended this place and she was right! reminds me of childhood comfort food. portions are huge and prices are fair. will definitely be back.",
            "tried the new location and it's much better than the original. seems like they learned from their mistakes. parking is still terrible though lol",
            "not sure what the fuss is about... maybe I ordered the wrong thing? the pasta was overcooked and sauce was too salty for my taste. my friend loved her dish though 🤷‍♀️",
            "perfect for a quick lunch but wouldn't bring a date here. food is good and fast but atmosphere is very casual. great value for money though!",
            "delivery was surprisingly fast and food was still hot! packaging was good too - nothing spilled. definitely ordering again when I'm too lazy to cook 😅",
            "went here for my birthday dinner and the server surprised me with a free dessert! small touches like that really matter. food was great too.",
            "this used to be my go-to spot but prices have gone up significantly. food is still good but not sure if it's worth the premium anymore.",
            "trendy instagram spot but food actually lives up to the hype! yes it's pricey but the presentation and taste are both on point. great for photos 📸",
            "hole in the wall place that doesn't look like much but the food is incredible! best kept secret in the neighborhood. cash only though.",
        ]
        
        # Generate reviews with realistic patterns
        reviews_data = []
        
        # Add the base authentic reviews
        for i, review_text in enumerate(authentic_reviews):
            restaurant = random.choice(self.restaurants)
            rating = self._get_realistic_rating(review_text.lower())
            
            reviews_data.append({
                'review_text': review_text,
                'rating': rating,
                'restaurant_name': restaurant,
                'cuisine_type': self.cuisines.get(restaurant, 'American'),
                'review_date': self._generate_realistic_date(),
                'reviewer_id': f"user_{random.randint(1000, 9999)}",
                'helpful_votes': random.randint(0, 50) if rating >= 4 else random.randint(0, 10)
            })
        
        # Add more reviews for popular restaurants (realistic pattern)
        popular_restaurants = random.sample(self.restaurants, 8)
        for restaurant in popular_restaurants:
            for _ in range(random.randint(3, 8)):
                review_text = self._generate_follow_up_review(restaurant)
                rating = self._get_realistic_rating(review_text.lower())
                
                reviews_data.append({
                    'review_text': review_text,
                    'rating': rating,
                    'restaurant_name': restaurant,
                    'cuisine_type': self.cuisines.get(restaurant, 'American'),
                    'review_date': self._generate_realistic_date(),
                    'reviewer_id': f"user_{random.randint(1000, 9999)}",
                    'helpful_votes': random.randint(0, 25)
                })
        
        return pd.DataFrame(reviews_data)
    
    def _get_realistic_rating(self, review_sentiment):
        """Generate ratings that match human behavior patterns"""
        negative_words = ['terrible', 'awful', 'worst', 'avoid', 'disgusting', 'horrible']
        positive_words = ['amazing', 'incredible', 'perfect', 'love', 'best', 'outstanding']
        mixed_words = ['decent', 'okay', 'meh', 'average', 'fine']
        
        if any(word in review_sentiment for word in negative_words):
            return random.choice([1, 1, 1, 2])  # Heavily weighted toward 1
        elif any(word in review_sentiment for word in positive_words):
            return random.choice([4, 5, 5, 5])  # Heavily weighted toward 5
        elif any(word in review_sentiment for word in mixed_words):
            return random.choice([2, 3, 3, 3, 4])  # Centered around 3
        else:
            return random.choice([2, 3, 3, 4, 4, 5])  # Slight positive bias
    
    def _generate_realistic_date(self):
        """Generate dates with realistic patterns (more recent reviews)"""
        # 60% of reviews from last 6 months, 30% from 6-12 months, 10% older
        rand = random.random()
        if rand < 0.6:
            days_ago = random.randint(1, 180)
        elif rand < 0.9:
            days_ago = random.randint(180, 365)
        else:
            days_ago = random.randint(365, 730)
        
        return (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    def _generate_follow_up_review(self, restaurant_name):
        """Generate follow-up reviews for popular restaurants"""
        templates = [
            f"Back at {restaurant_name} again! Still consistently good.",
            f"Third time here this month... yeah I'm addicted 😅",
            f"Brought my parents to {restaurant_name} and they loved it!",
            f"Regular customer here. Quality has been consistent.",
            f"Another great meal at {restaurant_name}. Never disappoints!",
            f"Tried a different dish this time at {restaurant_name}. Also excellent!",
            f"Date night at {restaurant_name} was perfect as always.",
            f"Quick lunch at {restaurant_name}. Fast service as usual.",
            f"Celebrating here again because we know it's reliable.",
            f"Recommended {restaurant_name} to friends. They thanked me!"
        ]
        return random.choice(templates)
    
    def add_realistic_data_issues(self, df):
        """Add realistic data quality issues"""
        df_copy = df.copy()
        
        # Add some missing ratings (people forget to rate)
        missing_indices = random.sample(range(len(df_copy)), k=max(1, len(df_copy)//25))
        df_copy.loc[missing_indices, 'rating'] = np.nan
        
        # Add restaurant name variations (typos/inconsistencies)
        name_variants = {
            "Tony's Corner Pizza": ["Tonys Corner Pizza", "Tony's Pizza Corner", "Tony Corner Pizza"],
            "Blue Moon Cafe": ["BlueMoon Cafe", "Blue Moon Coffee", "Blue Moon Café"],
            "The Rusty Anchor": ["Rusty Anchor", "The Rusty Anchor Bar", "Rusty Anchor Restaurant"]
        }
        
        for original, variants in name_variants.items():
            mask = df_copy['restaurant_name'] == original
            if mask.any():
                variant_indices = df_copy[mask].sample(n=min(2, mask.sum())).index
                for idx in variant_indices:
                    df_copy.loc[idx, 'restaurant_name'] = random.choice(variants)
        
        # Add duplicate review (accidental double submission)
        if len(df_copy) > 5:
            duplicate_idx = random.choice(range(len(df_copy)))
            duplicate_row = df_copy.iloc[duplicate_idx].copy()
            duplicate_row['reviewer_id'] = df_copy.iloc[duplicate_idx]['reviewer_id']  # Same user
            df_copy = pd.concat([df_copy, duplicate_row.to_frame().T], ignore_index=True)
        
        return df_copy

def main():
    """Generate authentic restaurant review dataset"""
    print("🏪 Generating Authentic Restaurant Review Dataset...")
    print("=" * 60)
    
    # Initialize generator
    generator = AuthenticDataGenerator()
    
    # Generate reviews
    df = generator.generate_authentic_reviews()
    
    # Add realistic data issues
    df = generator.add_realistic_data_issues(df)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save main dataset
    df.to_csv('data/restaurant_reviews.csv', index=False)
    
    # Save metadata
    metadata = {
        'total_reviews': len(df),
        'date_generated': datetime.now().isoformat(),
        'restaurants': len(df['restaurant_name'].unique()),
        'cuisines': df['cuisine_type'].unique().tolist(),
        'rating_distribution': df['rating'].value_counts().to_dict(),
        'data_quality_notes': [
            'Contains realistic typos and inconsistencies',
            'Includes duplicate reviews and missing ratings',
            'Restaurant name variations included',
            'Dates weighted toward recent reviews'
        ]
    }
    
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"✅ Generated {len(df)} authentic reviews")
    print(f"🏭 {len(df['restaurant_name'].unique())} unique restaurants")
    print(f"🍽️ {len(df['cuisine_type'].unique())} cuisine types")
    
    print(f"\n📊 Rating Distribution:")
    rating_counts = df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        if not pd.isna(rating):
            percentage = (count / len(df)) * 100
            print(f"   {int(rating)}⭐: {count:3d} reviews ({percentage:5.1f}%)")
    
    missing_ratings = df['rating'].isna().sum()
    if missing_ratings > 0:
        print(f"   Missing: {missing_ratings} reviews ({(missing_ratings/len(df))*100:.1f}%)")
    
    print(f"\n🗓️ Date Range:")
    print(f"   Earliest: {df['review_date'].min()}")
    print(f"   Latest: {df['review_date'].max()}")
    
    print(f"\n💾 Files saved:")
    print(f"   📄 data/restaurant_reviews.csv")
    print(f"   📋 data/dataset_metadata.json")
    
    print(f"\n📝 Sample reviews:")
    for i, row in df.head(3).iterrows():
        stars = '⭐' * int(row['rating']) if not pd.isna(row['rating']) else '❓'
        print(f"\n{stars} {row['restaurant_name']} ({row['cuisine_type']})")
        print(f"   \"{row['review_text'][:100]}{'...' if len(row['review_text']) > 100 else ''}\"")

if __name__ == "__main__":
    main()