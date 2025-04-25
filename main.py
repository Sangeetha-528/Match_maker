import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("extended_dating_app_user_profiles.csv")
    df = df.fillna("").astype(str)
    profile_texts = df.apply(lambda row: ' | '.join([f"{col}: {row[col]}" for col in row.index]), axis=1)
    return df, profile_texts
# Get embeddings using OpenAI API
@st.cache_data(show_spinner="Generating profile embeddings...")
def get_all_embeddings(texts):
    return np.array([
        client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-ada-002").data[0].embedding
        for text in texts
    ])
# Embed a single profile
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
# UI
st.title(":cupid: VFE Matchmaker")
st.markdown("Fill out the form below to find your top 5 soulmate matches based on your thoughts and beliefs!")
# Profile input form
with st.form("profile_form"):
    user_input = {
        "user_id": "9999",
        "family_meaning": st.text_area("What does family mean to you?"),
        "favorite_holiday": st.text_area("What is your favorite holiday destination?"),
        "self_description": st.text_area("How would you describe yourself?"),
        "marriage_meaning": st.text_area("What does marriage mean to you?"),
        "relationship_importance": st.text_area("What's important to you in a relationship?"),
        "friend_description": st.text_area("How would your friends describe you?"),
        "knows_best": st.text_area("Who would you say knows you the best?"),
        "happiest_memory": st.text_area("What is your happiest memory?"),
        "memorable_place": st.text_area("Most memorable place you’ve visited? What made it unforgettable?"),
        "biggest_regret": st.text_area("What is your biggest regret?"),
        "investment_opinion": st.text_area("Best financial investments, according to you?"),
        "anything_else": st.text_area("Anything else about yourself or life partner that you feel is important?"),
        "physical_dislike": st.text_area("A physical attribute about yourself that you dislike?"),
    }
    submitted = st.form_submit_button("Find Matches")
# Load and process data
df, profile_texts = load_data()
profile_embeddings = get_all_embeddings(profile_texts)
# Matchmaking logic
if submitted:
    with st.spinner("Finding your top 5 matches..."):
        # Create test profile text
        test_profile_text = ' | '.join([f"{k}: {v}" for k, v in user_input.items()])
        test_embedding = np.array(get_embedding(test_profile_text)).reshape(1, -1)
        # Compute cosine similarity
        cosine_scores = cosine_similarity(test_embedding, profile_embeddings)[0]
        top_indices = np.argsort(cosine_scores)[-5:][::-1]
        top_scores = cosine_scores[top_indices]
        # Show top 5 matches
        st.subheader(":mag: Top 5 Profile Matches")
        for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            matched_row = df.iloc[idx]
            with st.expander(f"{i}. User ID: {matched_row['user_id']} | Score: {score:.4f}"):
                for col in matched_row.index:
                    st.markdown(f"**{col.replace('_', ' ').capitalize()}**: {matched_row[col]}")









