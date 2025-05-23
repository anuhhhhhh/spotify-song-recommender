
import streamlit as st
from recommender import build_dataset, cluster_and_recommend

st.title("🎧 Spotify Song Recommender")
st.write("Select a song you like and get 5 recommendations based on audio features.")

seed_songs = ["Peaches", "Blinding Lights", "Stay", "SICKO MODE", "As It Was", "Bad Habit", "Save Your Tears", "HUMBLE.", "Uptown Funk", "Watermelon Sugar"]

df = build_dataset(seed_songs)

input_song = st.selectbox("🎵 Pick a song:", df['name'].tolist())

if st.button("🔍 Recommend Similar Songs"):
    recs = cluster_and_recommend(df, input_song)

    if recs.empty:
        st.write("No recommendations found. Try another song.")
    else:
        st.subheader("🎶 Recommendations:")
        for _, row in recs.iterrows():
            st.write(f"**{row['name']}** by *{row['artist']}*")
