import streamlit as st
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.set_page_config(page_title="Hotel Recommendation System", layout="wide")

st.title("🏨 Hotel Recommendation System")
st.write("Get hotel recommendations based on **location** and **trip description** using NLP.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("E:/MY_Projects/Hotel Recomendation/Hotel_Reviews.csv")
    df.Hotel_Address = df.Hotel_Address.str.replace("United Kingdom", "INDIA")
    df["countries"] = df.Hotel_Address.apply(lambda x: x.split(" ")[-1].lower())

    df.drop([
        'Additional_Number_of_Scoring', 'Review_Date',
        'Reviewer_Nationality', 'Negative_Review',
        'Review_Total_Negative_Word_Counts',
        'Total_Number_of_Reviews', 'Positive_Review',
        'Review_Total_Positive_Word_Counts',
        'Total_Number_of_Reviews_Reviewer_Has_Given',
        'Reviewer_Score', 'days_since_review',
        'lat', 'lng'
    ], axis=1, inplace=True)

    def impute(column):
        column = column[0]
        if type(column) != list:
            return "".join(literal_eval(column))
        return column

    df["Tags"] = df[["Tags"]].apply(impute, axis=1)
    df["Tags"] = df["Tags"].str.lower()

    return df

data = load_data()

# Recommendation function
def recommend_hotel(location, description):
    tokens = word_tokenize(description.lower())
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()

    user_set = set(lemm.lemmatize(w) for w in tokens if w not in stop_words)

    country_df = data[data['countries'] == location.lower()].reset_index(drop=True)

    similarity = []

    for i in range(country_df.shape[0]):
        tag_tokens = word_tokenize(country_df["Tags"][i])
        tag_set = set(lemm.lemmatize(w) for w in tag_tokens if w not in stop_words)
        similarity.append(len(tag_set.intersection(user_set)))

    country_df['similarity'] = similarity

    country_df = country_df.sort_values(
        ['similarity', 'Average_Score'],
        ascending=False
    ).drop_duplicates('Hotel_Name')

    return country_df[['Hotel_Name', 'Average_Score', 'similarity', 'Hotel_Address']].head(5)

# Sidebar inputs
st.sidebar.header("🔍 Search Filters")
location = st.sidebar.text_input("Enter Country", "italy")
description = st.sidebar.text_area(
    "Describe your trip",
    "business trip near city center"
)

if st.sidebar.button("Recommend Hotels"):
    results = recommend_hotel(location, description)

    if results.empty:
        st.warning("No hotels found for this location.")
    else:
        st.subheader("🏆 Top 5 Recommended Hotels")
        st.dataframe(results, use_container_width=True)

        # Visualization
        st.subheader("📊 Hotel Scores Visualization")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(results['Hotel_Name'], results['Average_Score'])

        for i, val in enumerate(results['similarity']):
            ax.text(results['Average_Score'].iloc[i] + 0.02, i,
                    f"Similarity: {val}", va='center')

        ax.set_xlabel("Average Score")
        ax.set_ylabel("Hotel Name")
        ax.invert_yaxis()
        st.pyplot(fig)
