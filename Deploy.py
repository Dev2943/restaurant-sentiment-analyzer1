import os

print("🔧 Creating configuration files...")

# Create requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("""streamlit==1.28.0
pandas==2.1.1
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
wordcloud==1.9.2
nltk==3.8.1
textblob==0.17.1
scikit-learn==1.3.0
""")

# Create .streamlit directory and config
os.makedirs('.streamlit', exist_ok=True)
with open('.streamlit/config.toml', 'w') as f:
    f.write("""[theme]
primaryColor = "#2E8B57"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
""")

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("✅ All files created successfully!")
print("✅ requirements.txt - lists all packages to install")
print("✅ .streamlit/config.toml - app styling")
print("✅ data/ - folder for your dataset")
print("✅ models/ - folder for AI models")
print("\n🚀 Next step: pip install -r requirements.txt")