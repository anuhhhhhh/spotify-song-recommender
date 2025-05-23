# recommender.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from spotify_api import get_spotify_client

sp = get_spotify_client()

def get_audio_features(track_name, artist_name=None):
    query = f"{track_name} {artist_name or ''}"
    results = sp.search(q=query, limit=1, type='track')
    
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']
        features = sp.audio_features([track_id])[0]
        features['name'] = track['name']
        features['artist'] = track['artists'][0]['name']
        return features
    else:
        return None

def build_dataset(seed_tracks):
    data = []
    for song in seed_tracks:
        features = get_audio_features(song)
        if features:
            data.append(features)
    
    df = pd.DataFrame(data)
    return df

def cluster_and_recommend(df, input_song, n_clusters=5):
    feature_cols = ['danceability', 'energy', 'tempo', 'valence', 'liveness']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    try:
        input_cluster = df[df['name'].str.lower() == input_song.lower()]['cluster'].values[0]
        recs = df[(df['cluster'] == input_cluster) & (df['name'].str.lower() != input_song.lower())]
        return recs.sample(min(5, len(recs)))  # return up to 5 recs
    except IndexError:
        return pd.DataFrame()
