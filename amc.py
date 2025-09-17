import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Amazon Music Recommendation Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Amazon Music theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #FF9500 0%, #FF6B35 50%, #232F3E 100%);
        color: white;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        background: linear-gradient(135deg, #FF9500 0%, #FF6B35 50%, #232F3E 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #232F3E 0%, #37475A 100%);
        color: white;
    }
    .Widget>label {
        color: white;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #37475A;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FF9500;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the music dataset"""
    try:
        # You'll need to replace this with your actual data loading
        # For demo purposes, creating sample data structure
        return pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(1000)],
            'track_name': [f'Song {i}' for i in range(1000)],
            'artist_name': [f'Artist {i%50}' for i in range(1000)],
            'danceability': np.random.uniform(0, 1, 1000),
            'energy': np.random.uniform(0, 1, 1000),
            'loudness': np.random.uniform(-60, 0, 1000),
            'speechiness': np.random.uniform(0, 1, 1000),
            'acousticness': np.random.uniform(0, 1, 1000),
            'instrumentalness': np.random.uniform(0, 1, 1000),
            'liveness': np.random.uniform(0, 1, 1000),
            'valence': np.random.uniform(0, 1, 1000),
            'tempo': np.random.uniform(60, 200, 1000),
            'duration_ms': np.random.uniform(180000, 300000, 1000)
        })
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df, audio_features):
    """Preprocess the data for clustering"""
    # Remove duplicates and handle missing values
    df_clean = df.drop_duplicates().dropna()
    
    # Extract features and normalize
    features_df = df_clean[audio_features]
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_df)
    
    return df_clean, normalized_features, scaler

@st.cache_data
def perform_clustering(normalized_features, method='kmeans', **kwargs):
    """Perform clustering based on selected method"""
    if method == 'kmeans':
        k = kwargs.get('k', 5)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(normalized_features)
        return labels, model
    
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(normalized_features)
        return labels, model
    
    elif method == 'hierarchical':
        n_clusters = kwargs.get('n_clusters', 5)
        linkage_method = kwargs.get('linkage_method', 'ward')
        linkage_matrix = linkage(normalized_features, method=linkage_method)
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return labels, linkage_matrix

@st.cache_data
def find_optimal_k(normalized_features, max_k=15):
    """Find optimal k using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(normalized_features, labels))
    
    return k_range, inertias, silhouette_scores

def create_pca_visualization(normalized_features, labels, title):
    """Create PCA visualization of clusters"""
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(normalized_features)
    
    fig = px.scatter(
        x=pca_features[:, 0], y=pca_features[:, 1],
        color=labels.astype(str),
        title=title,
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def recommend_songs(df, target_song_index, n_recommendations=5):
    """Simple content-based recommendation system"""
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df[audio_features])
    
    # Find similar songs using KNN
    knn = NearestNeighbors(n_neighbors=n_recommendations+1, metric='cosine')
    knn.fit(features_normalized)
    
    distances, indices = knn.kneighbors([features_normalized[target_song_index]])
    
    # Exclude the target song itself
    recommended_indices = indices[0][1:]
    recommended_songs = df.iloc[recommended_indices]
    
    return recommended_songs

# Main application
def main():
    # Sidebar navigation
    st.sidebar.title("🎵 Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Home", "Clustering Analysis", "Music Recommendation", "About Developer"])
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check your dataset.")
        return
    
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    if page == "Home":
        home_page()
    elif page == "Clustering Analysis":
        clustering_page(df, audio_features)
    elif page == "Music Recommendation":
        recommendation_page(df, audio_features)
    elif page == "About Developer":
        about_page()

def home_page():
    """Home page with project overview"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">
            🎵 Amazon Music Recommendation Dashboard
        </h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">
            Discover music patterns through advanced clustering techniques and personalized recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Project Objective</h3>
            <p>Automatically group similar songs based on their audio characteristics using unsupervised machine learning techniques, without relying on manual genre labels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset Features</h3>
            <ul>
                <li><strong>Danceability:</strong> How suitable a track is for dancing</li>
                <li><strong>Energy:</strong> Perceptual measure of intensity and power</li>
                <li><strong>Valence:</strong> Musical positiveness conveyed by a track</li>
                <li><strong>Tempo:</strong> Overall estimated tempo in BPM</li>
                <li><strong>And more...</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Clustering methods overview
    st.markdown("### 🔬 Clustering Methods Implemented")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>K-Means Clustering</h4>
            <p><strong>Approach:</strong> Partitional clustering</p>
            <p><strong>Best for:</strong> Spherical clusters</p>
            <p><strong>Optimization:</strong> Elbow method + Silhouette score</p>
            <p><strong>Advantage:</strong> Simple and efficient</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>DBSCAN Clustering</h4>
            <p><strong>Approach:</strong> Density-based clustering</p>
            <p><strong>Best for:</strong> Arbitrary shaped clusters</p>
            <p><strong>Parameters:</strong> eps (neighborhood size), min_samples</p>
            <p><strong>Advantage:</strong> Identifies noise and outliers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Hierarchical Clustering</h4>
            <p><strong>Approach:</strong> Agglomerative clustering</p>
            <p><strong>Best for:</strong> Nested cluster structures</p>
            <p><strong>Visualization:</strong> Dendrogram</p>
            <p><strong>Advantage:</strong> No need to specify cluster count initially</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation system overview
    st.markdown("### 🎼 Music Recommendation System")
    st.markdown("""
    <div class="metric-card">
        <p>Our content-based recommendation system analyzes audio features to find songs with similar characteristics. 
        It uses cosine similarity to identify tracks that share similar musical properties like tempo, energy, and mood.</p>
    </div>
    """, unsafe_allow_html=True)

def clustering_page(df, audio_features):
    """Clustering analysis page"""
    st.title("🔬 Clustering Analysis")
    
    # Preprocess data
    df_clean, normalized_features, scaler = preprocess_data(df, audio_features)
    
    # Sidebar controls
    st.sidebar.markdown("### Clustering Parameters")
    
    # Method selection
    method = st.sidebar.selectbox("Select Clustering Method:", 
                                 ["K-Means", "DBSCAN", "Hierarchical"])
    
    if method == "K-Means":
        # K-Means specific controls
        auto_k = st.sidebar.checkbox("Auto-detect optimal k", value=True)
        
        if auto_k:
            k_range, inertias, silhouette_scores = find_optimal_k(normalized_features)
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            # Display elbow method results
            col1, col2 = st.columns(2)
            
            with col1:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, 
                                             mode='lines+markers', name='WCSS'))
                fig_elbow.update_layout(title="Elbow Method", 
                                      xaxis_title="Number of Clusters (k)",
                                      yaxis_title="WCSS (Inertia)",
                                      plot_bgcolor='rgba(0,0,0,0)',
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      font_color='white')
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, 
                                           mode='lines+markers', name='Silhouette Score'))
                fig_sil.update_layout(title="Silhouette Score Analysis", 
                                    xaxis_title="Number of Clusters (k)",
                                    yaxis_title="Silhouette Score",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white')
                st.plotly_chart(fig_sil, use_container_width=True)
            
            k = optimal_k
            st.sidebar.success(f"Optimal k detected: {k}")
        else:
            k = st.sidebar.slider("Number of clusters (k):", 2, 15, 5)
        
        # Perform K-Means clustering
        labels, model = perform_clustering(normalized_features, 'kmeans', k=k)
        
        # Display results
        st.subheader(f"K-Means Clustering Results (k={k})")
        
        # Metrics
        silhouette_avg = silhouette_score(normalized_features, labels)
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Clusters", k)
        col2.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        col3.metric("Inertia", f"{model.inertia_:.2f}")
        
    elif method == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps):", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Minimum samples:", 2, 20, 5)
        
        # Perform DBSCAN clustering
        labels, model = perform_clustering(normalized_features, 'dbscan', 
                                         eps=eps, min_samples=min_samples)
        
        # Display results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        st.subheader(f"DBSCAN Clustering Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Clusters", n_clusters)
        col2.metric("Noise Points", n_noise)
        
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(normalized_features[mask], labels[mask])
                col3.metric("Silhouette Score", f"{silhouette_avg:.3f}")
    
    elif method == "Hierarchical":
        n_clusters = st.sidebar.slider("Number of clusters:", 2, 15, 5)
        linkage_method = st.sidebar.selectbox("Linkage method:", 
                                            ["ward", "complete", "average", "single"])
        
        # Perform Hierarchical clustering
        labels, linkage_matrix = perform_clustering(normalized_features, 'hierarchical',
                                                  n_clusters=n_clusters, 
                                                  linkage_method=linkage_method)
        
        # Display results
        st.subheader(f"Hierarchical Clustering Results")
        
        silhouette_avg = silhouette_score(normalized_features, labels)
        col1, col2 = st.columns(2)
        col1.metric("Number of Clusters", n_clusters)
        col2.metric("Silhouette Score", f"{silhouette_avg:.3f}")
    
    # PCA Visualization
    st.subheader("Cluster Visualization (PCA)")
    fig_pca = create_pca_visualization(normalized_features, labels, 
                                     f"{method} Clustering Results")
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    df_with_clusters = df_clean.copy()
    df_with_clusters['cluster'] = labels
    
    cluster_stats = df_with_clusters.groupby('cluster')[audio_features].mean()
    
    # Heatmap of cluster characteristics
    fig_heatmap = px.imshow(cluster_stats.T, 
                           labels=dict(x="Cluster", y="Features", color="Average Value"),
                           title="Cluster Characteristics Heatmap",
                           color_continuous_scale='Viridis')
    fig_heatmap.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)',
                             font_color='white')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Cluster statistics table
    st.dataframe(cluster_stats.round(3))

def recommendation_page(df, audio_features):
    """Music recommendation page"""
    st.title("🎼 Music Recommendation System")
    
    st.markdown("""
    <div class="metric-card">
        <p>Select a song from the dropdown below to get personalized recommendations based on audio similarity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Song selection
    song_options = [f"{row['track_name']} - {row['artist_name']}" 
                   for _, row in df.iterrows()]
    selected_song = st.selectbox("Choose a song:", song_options)
    
    # Number of recommendations
    n_recommendations = st.slider("Number of recommendations:", 1, 20, 5)
    
    if st.button("Get Recommendations"):
        # Find selected song index
        selected_index = song_options.index(selected_song)
        
        # Get recommendations
        recommended_songs = recommend_songs(df, selected_index, n_recommendations)
        
        # Display selected song details
        st.subheader("Selected Song")
        selected_song_data = df.iloc[selected_index]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Track:** {selected_song_data['track_name']}")
            st.write(f"**Artist:** {selected_song_data['artist_name']}")
        
        with col2:
            # Audio features radar chart
            features_values = [selected_song_data[feature] for feature in audio_features[:6]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=features_values,
                theta=audio_features[:6],
                fill='toself',
                name='Selected Song'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="Audio Features Profile",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Display recommendations
        st.subheader("Recommended Songs")
        
        for i, (_, song) in enumerate(recommended_songs.iterrows(), 1):
            with st.expander(f"{i}. {song['track_name']} - {song['artist_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Audio Features:**")
                    for feature in audio_features[:5]:
                        st.write(f"- {feature.capitalize()}: {song[feature]:.3f}")
                
                with col2:
                    st.write("**Additional Info:**")
                    for feature in audio_features[5:]:
                        st.write(f"- {feature.capitalize()}: {song[feature]:.3f}")

def about_page():
    """About developer page"""
    st.title("👨‍💻 About Developer")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div class="metric-card">
            <h2>Amazon Music Clustering Project</h2>
            <p style="font-size: 1.1rem;">
                This project demonstrates the application of unsupervised machine learning techniques 
                for music analysis and recommendation systems.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🛠️ Technical Stack</h3>
            <ul>
                <li><strong>Python:</strong> Core programming language</li>
                <li><strong>Streamlit:</strong> Web application framework</li>
                <li><strong>Scikit-learn:</strong> Machine learning algorithms</li>
                <li><strong>Pandas & NumPy:</strong> Data manipulation</li>
                <li><strong>Plotly:</strong> Interactive visualizations</li>
                <li><strong>Matplotlib & Seaborn:</strong> Statistical plotting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Key Features</h3>
            <ul>
                <li>Multiple clustering algorithms comparison</li>
                <li>Interactive parameter tuning</li>
                <li>PCA visualization for cluster analysis</li>
                <li>Content-based recommendation system</li>
                <li>Real-time clustering performance metrics</li>
                <li>Responsive Amazon Music themed UI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Project Objectives Achieved</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div>
                <h4>✅ Data Exploration & Cleaning</h4>
                <p>Comprehensive analysis of audio features with proper preprocessing</p>
            </div>
            <div>
                <h4>✅ Multiple Clustering Methods</h4>
                <p>Implementation of K-Means, DBSCAN, and Hierarchical clustering</p>
            </div>
            <div>
                <h4>✅ Model Optimization</h4>
                <p>Elbow method and silhouette analysis for optimal parameters</p>
            </div>
            <div>
                <h4>✅ Interactive Visualization</h4>
                <p>PCA plots and cluster characteristic analysis</p>
            </div>
            <div>
                <h4>✅ Recommendation System</h4>
                <p>Content-based filtering using audio feature similarity</p>
            </div>
            <div>
                <h4>✅ User Interface</h4>
                <p>Professional Streamlit dashboard with Amazon Music theme</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div class="metric-card">
            <h3>🚀 Future Enhancements</h3>
            <p>Potential improvements could include deep learning-based embeddings, 
            real-time streaming integration, and collaborative filtering approaches.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()