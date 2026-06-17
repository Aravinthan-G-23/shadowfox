import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the data
df = pd.read_csv("X data.csv")

# Data Preprocessing
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

# Perform Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    if scores['compound'] > 0.05:
        return 'Positive'
    elif scores['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(get_sentiment)

st.title('Sentiment Analysis Dashboard')
st.write('This is a simple Streamlit app to display sentiment analysis results.')

# Display the sentiment distribution from the DataFrame
st.subheader('Sentiment Distribution')
sentiment_counts = df['sentiment'].value_counts()

# Use matplotlib and st.pyplot to display the bar chart
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
sentiment_counts.plot(kind='bar', ax=ax_bar, color=['skyblue', 'lightcoral', 'lightgreen'])
ax_bar.set_title('Sentiment Distribution')
ax_bar.set_xlabel('Sentiment')
ax_bar.set_ylabel('Count')
ax_bar.tick_params(axis='x', rotation=45)
st.pyplot(fig_bar)
plt.close(fig_bar) # Close the figure to free up memory

st.subheader('Sentiment Percentage')
sentiment_percent = (df['sentiment'].value_counts(normalize=True) * 100).round(2)

# Use matplotlib and st.pyplot to display the sentiment percentage table
fig_table, ax_table = plt.subplots(figsize=(6, 2))
ax_table.axis('off') # Hide axes
ax_table.axis('tight') # Adjust layout
table_data = sentiment_percent.reset_index()
table_data.columns = ['Sentiment', 'Percentage']
table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
st.pyplot(fig_table)
plt.close(fig_table) # Close the figure to free up memory

st.subheader('Word Cloud for Positive Sentiments')

positive_text = " ".join(df[df['sentiment']=="Positive"]['clean_text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(10,5))
ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
ax_wordcloud.axis('off')
st.pyplot(fig_wordcloud)
plt.close(fig_wordcloud)
