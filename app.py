from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Flask app
app = Flask(__name__)

# Load your dataset
df = pd.read_csv('enhanced_music_data.csv')

numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

# --------- Existing Feature-Based Recommendation ----------
def recommend_songs(song_name, df, num_recommendations=5):
    try:
        # Handle case-insensitive match
        original_song_row = df[df["name"].str.lower() == song_name.lower()]
        if original_song_row.empty:
            return pd.DataFrame([{"name": "Error", "artists": "Song not found", "year": ""}])

        song_cluster = original_song_row["Cluster"].values[0]

        # Get all songs in the same cluster, and reset index for safe indexing
        same_cluster_songs = df[df["Cluster"] == song_cluster].reset_index(drop=True)

        # Now get the index of the input song **within same_cluster_songs**
        # We match by song name (case-insensitive) again
        matched_song_row = same_cluster_songs[same_cluster_songs["name"].str.lower() == song_name.lower()]
        if matched_song_row.empty:
            return pd.DataFrame([{"name": "Error", "artists": "Song not found in cluster", "year": ""}])

        # This index is guaranteed to be within similarity matrix
        song_index = matched_song_row.index[0]

        # Compute cosine similarity matrix for this cluster
        cluster_features = same_cluster_songs[numerical_features]
        similarity = cosine_similarity(cluster_features)

        # Get most similar songs (excluding the song itself)
        similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
        recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]

        return recommendations

    except Exception as e:
        print("Error in recommend_songs():", e)
        return pd.DataFrame([{"name": "Error", "artists": str(e), "year": ""}])


# --------- NEW Mood-Based Recommendation ----------
def recommend_songs_by_mood(df, mood_input, num_recommendations=5):
    mood_matches = df[df['mood'] == mood_input]
    if len(mood_matches) >= num_recommendations:
        recommendations = mood_matches.sample(n=num_recommendations, random_state=42)
    else:
        recommendations = mood_matches
    return recommendations[["name", "year", "artists"]]


def recommend_popular_songs(df, num_recommendations=5):
    if "popularity" not in df.columns:
        return pd.DataFrame([{"name": "Error", "artists": "No popularity data found", "year": ""}])

    top_songs = df.sort_values(by="popularity", ascending=False).head(num_recommendations)
    return top_songs[["name", "year", "artists"]]


# --------- Routes ----------
@app.route("/")
def index():
    return render_template('index.html')

''''@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        song_name = request.form.get("song_name")
        mood_input = request.form.get("mood")

        if song_name:
            recommendations = recommend_songs(song_name, df).to_dict(orient="records")
        elif mood_input:
            recommendations = recommend_songs_by_mood(df, mood_input.lower(), num_recommendations=5).to_dict(orient="records")
        else:
            recommendations = [{"name": "Error", "artists": "No input provided", "year": ""}]
    return render_template("index.html", recommendations=recommendations)
'''
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        song_name = request.form.get("song_name")
        mood_input = request.form.get("mood")
        popular_trigger = request.form.get("popular")

        if song_name:
            recommendations = recommend_songs(song_name, df).to_dict(orient="records")
        elif mood_input:
            recommendations = recommend_songs_by_mood(df, mood_input.lower(), num_recommendations=5).to_dict(orient="records")
        elif popular_trigger == "yes":
            recommendations = recommend_popular_songs(df).to_dict(orient="records")
        else:
            recommendations = [{"name": "Error", "artists": "No input provided", "year": ""}]

    return render_template("index.html", recommendations=recommendations)

@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_value = request.form.get("feedback_value")
    if feedback_value:
        with open("feedback.txt", "a") as f:
            f.write(f"{feedback_value} - submitted at {pd.Timestamp.now()}\n")
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
