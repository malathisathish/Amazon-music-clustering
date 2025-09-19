# Amazon Music Clustering & Recommendation Project

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Overview](#dataset-overview)
3. [Problem Statement](#problem-statement)
 4. [Clustering Analysis](#clustering-analysis)

   * 4.1 [K-Means Clustering](#k-means-clustering)
   * 4.2 [DBSCAN Clustering](#dbscan-clustering)
   * 4.3 [Hierarchical Clustering](#hierarchical-clustering)
5. [Recommendation System](#recommendation-system)
6. [Insights & Observations](#insights--observations)
7. [Business Implications](#business-implications)
8. [Project Objectives](#project-objectives)
9. [How to Run](#how-to-run)
10. [Tech Stack](#tech-stack)
11. [Future Enhancements](#future-enhancements)
12. [Acknowledgments](#-acknowledgments)
13. [Conclusion](#-conclusion)
14. [Author](#-author)

---

## Project Overview

The Amazon Music Clustering & Recommendation project explores and analyzes song-level and artist-level data from Amazon Music to generate meaningful clusters of music tracks and provide personalized recommendations. The project uses **machine learning clustering algorithms**, **content-based similarity**, and **interactive dashboards** to empower music discovery and curation.

---

## Dataset Overview

* **Dataset Name:** Amazon Music Clustering
* **Source:** Amazon Music platform
* **Format:** CSV
* **Records:** 95,837
* **Features:** 23 (7 categorical, 12 float numerical, 4 integer numerical)

**Columns :**

* id\_songs, name\_song, popularity\_songs, duration\_ms, explicit, id\_artists, release\_date
* danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time\_signature
* followers, genres, name\_artists, popularity\_artists

**Timeframe Covered:** Tracks from early 20th century to 2020.

**Purpose:** To study music similarity, genre trends, artist influence, and playlist generation.

---

## Problem Statement

### ⚠️ The Challenge

Music streaming platforms struggle to recommend tracks effectively due to large, diverse catalogs and changing user preferences. Identifying meaningful clusters of songs based on **audio features** and **artist metadata** can help build intelligent recommendation systems.

### 🎯 The Aim

* Group songs with similar audio characteristics using clustering algorithms.
* Recommend tracks from the same cluster to enhance playlist personalization.
* Identify emerging trends and patterns for strategic decisions in music curation and marketing.

---
## 🗂️ Folder Structure

amazon-music-clustering/
┣ 📁data
┃ ┗ single_genre_artists.csv
┃ ┗ final_cleaned_amazon_music.csv
┣ 📁notebooks
┃ ┗ amazon_music_clustering.ipynb
┣ 📁report
┃ ┗ report.txt
┣ 📁 amazon_music_clustering.py
┣ 📄 requirements.txt
┣ 📄 README.md

---
## 🏗️ Technical Architecture

♥ **Data Collection**: CSV dataset from Amazon Music

♥ **Preprocessing**: Feature scaling, outlier removal, encoding categorical columns

♥ **Clustering**: K-Means, DBSCAN, Hierarchical

♥ **Recommendation System**: Content-based filtering using cosine similarity

♥ **Visualization**: Cluster plots, heatmaps, and feature distributions

---
## Clustering Analysis

### 4.1 K-Means Clustering

* **Parameters:** n\_clusters = 4, random\_state = 42
* **Silhouette Score:** 0.556
* **Davies-Bouldin Index:** 0.687
* **Cluster Insights:**

  * Cluster 0: Calm, instrumental-heavy tracks
  * Cluster 1: High-energy dance tracks
  * Cluster 2: Acoustic/experimental
  * Cluster 3: Spoken word, upbeat

### 4.2 DBSCAN Clustering

* **Parameters:** eps = 0.8, min\_samples = 5
* **Clusters Detected:** 4, Noise points = 0
* **Silhouette Score:** 0.639
* **Davies-Bouldin Index:** 0.653
* **Cluster Insights:** Similar to K-Means but density-aware

### 4.3 Hierarchical Clustering

* **Linkage:** Ward, n\_clusters = 4
* **Silhouette Score:** 0.639
* **Davies-Bouldin Index:** 0.540
* **Cluster Insights:** Reveals nested relationships and hierarchy among clusters

---

## Recommendation System

* **Approach:** Content-based filtering using cosine similarity on audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
* **Example:** For a given song, recommend other tracks from the same cluster with similar audio characteristics

---

## Insights & Observations

* High-energy, high-valence clusters correspond to danceable tracks.
* Acoustic and instrumental-heavy clusters correspond to calm or experimental music.
* DBSCAN effectively detects outliers, though in this dataset noise = 0.
* Hierarchical clustering shows nested similarities, helping explore subgroups.
* Cluster-based recommendations enhance playlist diversity.

---

## Business Implications

* **Music Platforms:** Improve user engagement via cluster-based playlists
* **Marketing:** Target niche user segments with specific tracks or playlists
* **Artists:** Identify trends and gaps to release content strategically

---

## Project Objectives

* Analyze track similarities and differences using audio features
* Generate meaningful song clusters using K-Means, DBSCAN, and Hierarchical clustering
* Recommend songs to users using content-based similarity
* Visualize clusters and insights via interactive dashboards
* Support strategic decision-making for music curation and marketing

---

## How to Run

1. **Load the Amazon music csv file(single_genre_artists.csv )**
2. **Optional:** Create a virtual environment

```bash
python -m venv venv
# Activate
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run Dashboard**

```bash
streamlit run streamlit_app/amazon_music_dashboard.py
```

5. **Open in Browser:** [http://localhost:8501](http://localhost:8501)

---

## Tech Stack

* **Frontend:** Streamlit, Plotly, Seaborn, Matplotlib
* **Backend:** Python, Pandas, Scikit-learn, NumPy
* **ML Models:** K-Means, DBSCAN, Hierarchical Clustering
* **Visualization:** Interactive cluster plots, recommendation tables

---

## Future Enhancements

* Integrate user listening data for personalized recommendations
* Real-time cluster updates as new songs are added
* Predictive modeling for emerging trends in music
* Enhanced dashboards with richer interactivity and filters

---

## 🙏 Acknowledgments

* **Amazon Music Dataset** – for providing rich music track data
* **Scikit-learn & NumPy** – for clustering, feature scaling, and similarity computations
* **Pandas & Matplotlib/Seaborn/Plotly** – for data wrangling and visualization
* **Streamlit** – for building interactive dashboards
* **GUVI mentors** – for guidance and support throughout the project

---

## ✅ Conclusion

This project bridges the gap between music analytics and personalized recommendations by combining **clustering algorithms, content-based filtering, and data visualization**.

It empowers users and music platforms to:

* Discover tracks that match mood, genre, or energy level
* Create data-driven playlists
* Identify emerging artists and niche music preferences
* Make insights actionable for engagement and marketing

✨ “Data transforms music into insights — helping listeners and platforms explore smarter.”

---

## 👩‍💻 Author

**Malathi.y | Data Science Enthusiast 🎓**

💬 Feedback? Contributions? Questions? Let’s connect!
📧 Email: [malathisathish2228@gmail.com](mailto:malathisathish2228@gmail.com)
💻 GitHub: [https://github.com/malathisathish](https://github.com/malathisathish)

