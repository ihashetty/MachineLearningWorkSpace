import pandas as pd
import joblib

# ---------------- LOAD TRAINED MODEL ----------------
model = joblib.load("spotify_kmeans_pipeline.pkl")

# ---------------- LOAD ORIGINAL DATA ----------------
df = pd.read_csv("spotify.csv")

FEATURE_COLS = [
    'danceability',
    'energy',
    'valence',
    'tempo',
    'duration_ms',
    'popularity'
]

# ---------------- HANDLE MISSING VALUES ----------------
X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

# ---------------- PREDICT CLUSTERS ----------------
df["cluster"] = model.predict(X)

# ---------------- SAVE CLUSTERED DATA ----------------
df.to_csv("spotify_clustered.csv", index=False)

print("✅ spotify_clustered.csv created successfully!")