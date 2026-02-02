import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="🎵 Spotify Song Recommender", layout="wide")

st.title("🎵 Spotify Song Cluster & Recommendation System")

# ---------------- INPUT FORM ----------------
with st.form("song_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)

    with col2:
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.number_input("Tempo", 50, 200, 120)

    with col3:
        duration_ms = st.number_input("Duration (ms)", 50000, 400000, 200000)
        popularity = st.slider("Popularity", 0, 100, 50)

    submit = st.form_submit_button("🎯 Predict & Recommend")

# ---------------- PREDICTION + RECOMMEND ----------------
if submit:
    payload = {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms,
        "popularity": popularity
    }

    response = requests.post(
        f"{API_URL}/recommend?top_n=5",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()

        st.success(f"🎶 Mood Cluster: **{result['cluster_name']}**")

        st.subheader("🎧 Recommended Songs")
        st.table(pd.DataFrame(result["recommendations"]))
    else:
        st.error("Recommendation failed")

# ---------------- HISTORY ----------------
st.divider()
st.subheader("📊 Prediction History")

if st.button("Load History"):
    history = requests.get(f"{API_URL}/history").json()
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.info("No predictions yet")