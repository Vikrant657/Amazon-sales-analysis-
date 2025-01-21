# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'amazon.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst Five Rows:")
print(data.head())

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Handle missing values (example: fill rating_count with median)
data['rating_count'].fillna(data['rating_count'].median(), inplace=True)

# Handle inconsistent data types (e.g., discounted_price as float)
data['discounted_price'] = pd.to_numeric(data['discounted_price'], errors='coerce')
data['actual_price'] = pd.to_numeric(data['actual_price'], errors='coerce')
data['discount_percentage'] = pd.to_numeric(data['discount_percentage'], errors='coerce')

# Drop duplicates
data.drop_duplicates(inplace=True)

# Verify cleaning
print("\nData after Cleaning:")
print(data.info())

top_categories = data['category'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
plt.title('Top 10 Categories by Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Category')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['rating'], bins=10, kde=True, color='blue')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='actual_price', y='discount_percentage', data=data, hue='category', alpha=0.7)
plt.title('Price vs Discount Percentage')
plt.xlabel('Actual Price')
plt.ylabel('Discount Percentage')
plt.show()

top_rated_products = data.sort_values(by=['rating', 'rating_count'], ascending=False).head(10)
print("Top Rated Products:")
print(top_rated_products[['product_id', 'product_name', 'rating', 'rating_count']])


from wordcloud import WordCloud

review_titles = ' '.join(data['review_title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_titles)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Review Titles')
plt.show()

from textblob import TextBlob

# Sentiment analysis on review content
data['review_sentiment'] = data['review_content'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['review_sentiment'], bins=20, kde=True, color='green')
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

import plotly.express as px

# Interactive scatter plot of price vs rating
fig = px.scatter(data, x='actual_price', y='rating', color='category',
                 size='rating_count', hover_name='product_name',
                 title='Price vs Rating by Category')
fig.show()

# Summarize key findings
print("Summary of Key Insights:")
print(f"1. Total Products: {data['product_id'].nunique()}")
print(f"2. Average Rating: {data['rating'].mean():.2f}")
print(f"3. Top Category by Products: {data['category'].mode()[0]}")
print(f"4. Highest Discount: {data.loc[data['discount_percentage'].idxmax(), 'product_name']}")
