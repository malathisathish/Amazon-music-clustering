import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Amazon Music Recommendation Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Amazon Music-inspired theme with enhanced font visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #E6F0FA 0%, #B3D9FF 100%);
        color: #232F3E;
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #E6F0FA 0%, #B3D9FF 100%);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .sidebar .sidebar-content {
        background: #232F3E;
        color: #FFFFFF;
        border-right: 3px solid #00A8E1;
    }
    
    .sidebar .sidebar-content h2, .sidebar .sidebar-content label {
        color: #FFFFFF !important;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    h1, h2, h3, h4 {
        font-family: 'Roboto', sans-serif !important;
        color: #232F3E !important;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    p, li, div, span {
        font-family: 'Roboto', sans-serif;
        color: #37475A;
        line-height: 1.6;
        font-weight: 400;
        text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2);
    }
    
    .stSelectbox > div > div > div {
        color: #232F3E;
        background-color: #FFFFFF;
        border: 1px solid #00A8E1;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #FF9900;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00A8E1, #0077B6);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF9900, #F28C38);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E6E6E6;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        padding: 1rem;
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .music-note {
        animation: pulse 2s ease-in-out infinite;
        font-size: 1.5rem;
        margin: 0 5px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 5px;
        border: 1px solid #00A8E1;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #232F3E;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #00A8E1, #0077B6);
        color: #FFFFFF;
        font-weight: 700;
    }
    
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E6E6E6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #232F3E;
        font-family: 'Roboto', sans-serif;
    }
    
    .stDataFrame {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E6E6E6;
    }
</style>
""", unsafe_allow_html=True)

# Amazon Music Logo SVG
AMAZON_MUSIC_LOGO = """
<svg viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg" style="width: 300px; height: auto;">
    <defs>
        <linearGradient id="amazonGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#00A8E1;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FF9900;stop-opacity:1" />
        </linearGradient>
        <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </defs>
    <text x="10" y="25" font-family="Roboto, sans-serif" font-size="20" font-weight="bold" fill="#232F3E" filter="url(#glow)">amazon</text>
    <path d="M15 28 Q 65 38 115 28" stroke="#FF9900" stroke-width="2" fill="none" filter="url(#glow)"/>
    <circle cx="115" cy="28" r="2" fill="#FF9900"/>
    <text x="10" y="45" font-family="Roboto, sans-serif" font-size="18" font-weight="bold" fill="#232F3E" filter="url(#glow)">MUSIC</text>
    <circle cx="150" cy="30" r="8" fill="url(#amazonGradient)" filter="url(#glow)" class="music-note"/>
    <ellipse cx="158" cy="35" rx="4" ry="2" fill="url(#amazonGradient)"/>
    <line x1="158" y1="30" x2="158" y2="10" stroke="url(#amazonGradient)" stroke-width="2"/>
    <path d="M158 10 Q165 8 170 12" stroke="url(#amazonGradient)" stroke-width="2" fill="none"/>
    <path d="M175 25 Q180 20 185 25 Q190 30 195 25" stroke="#00A8E1" stroke-width="2" fill="none" opacity="0.8"/>
    <path d="M175 30 Q180 25 185 30 Q190 35 195 30" stroke="#FF9900" stroke-width="2" fill="none" opacity="0.8"/>
</svg>
"""
# Load example data for recommendation
@st.cache_data
def load_data():
    data = {
        'track_id': ['track_0', 'track_5386', 'track_1086', 'track_65445'],
        'track_name': ['Original Song', 'Recommended Song 1', 'Recommended Song 2', 'Recommended Song 3'],
        'artist_name': ['Artist A', 'Artist B', 'Artist C', 'Artist D'],
        'danceability': [0.568113, 0.732593, 0.516650, 0.576186],
        'energy': [0.183984, 0.049281, 0.310986, 0.383988],
        'loudness': [0.655572, 0.532349, 0.614779, 0.727039],
        'speechiness': [0.052893, 0.058368, 0.325413, 0.031612],
        'acousticness': [0.996988, 0.976908, 0.996988, 0.569277],
        'instrumentalness': [0.000016, 0.00000, 0.00183, 0.00000],
        'liveness': [0.325978, 0.406219, 0.263791, 0.141424],
        'valence': [0.654000, 0.729, 0.893, 0.287],
        'tempo': [0.554751, 0.571195, 0.778592, 0.481693],
        'duration_ms': [0.032345, 0.016900, 0.035578, 0.061452],
        'cluster': [3, 3, 3, 3]
    }
    return pd.DataFrame(data)

# Clustering data
kmeans_clusters = {
    'Cluster': ['0', '1', '2', '3'],
    'danceability': [0.481, 0.627, 0.539, 0.670],
    'energy': [0.379, 0.706, 0.374, 0.466],
    'loudness': [0.644, 0.765, 0.697, 0.661],
    'speechiness': [0.062, 0.079, 0.064, 0.889],
    'acousticness': [0.709, 0.182, 0.747, 0.595],
    'instrumentalness': [0.816, 0.032, 0.017, 0.001],
    'liveness': [0.187, 0.203, 0.189, 0.430],
    'valence': [0.437, 0.656, 0.489, 0.578],
    'tempo': [0.471, 0.519, 0.481, 0.418],
    'duration_ms': [0.046, 0.046, 0.043, 0.022]
}
kmeans_df = pd.DataFrame(kmeans_clusters).set_index('Cluster')

dbscan_clusters = {
    'Cluster': ['0', '1', '2', '3'],
    'danceability': [0.560, 0.629, 0.661, 0.524],
    'energy': [0.366, 0.706, 0.493, 0.380],
    'loudness': [0.700, 0.762, 0.669, 0.656],
    'speechiness': [0.059, 0.077, 0.879, 0.058],
    'acousticness': [0.744, 0.185, 0.537, 0.757],
    'instrumentalness': [0.018, 0.021, 0.000, 0.791],
    'liveness': [0.169, 0.208, 0.398, 0.197],
    'valence': [0.497, 0.666, 0.586, 0.498],
    'tempo': [0.485, 0.532, 0.445, 0.472],
    'duration_ms': [0.042, 0.047, 0.019, 0.040]
}
dbscan_df = pd.DataFrame(dbscan_clusters).set_index('Cluster')

hierarchical_clusters = {
    'Cluster': ['0', '1', '2', '3'],
    'danceability': [0.629, 0.560, 0.661, 0.524],
    'energy': [0.706, 0.366, 0.493, 0.380],
    'loudness': [0.762, 0.700, 0.669, 0.656],
    'speechiness': [0.077, 0.059, 0.879, 0.058],
    'acousticness': [0.185, 0.744, 0.537, 0.757],
    'instrumentalness': [0.021, 0.018, 0.000, 0.791],
    'liveness': [0.208, 0.169, 0.398, 0.197],
    'valence': [0.666, 0.497, 0.586, 0.498],
    'tempo': [0.532, 0.485, 0.445, 0.472],
    'duration_ms': [0.047, 0.042, 0.019, 0.040]
}
hierarchical_df = pd.DataFrame(hierarchical_clusters).set_index('Cluster')

# Main application
def main():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
        <h2>🎵 Amazon Music Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose a page:",
                               ["🏠 Home", "🔬 Clustering Analysis", "🎼 Music Recommendation", "📄 Project Report", "👨‍💻 About Developer"],
                               key="nav_selectbox")
    
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check your dataset.")
        return
    
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo']
    
    if page == "🏠 Home":
        home_page()
    elif page == "🔬 Clustering Analysis":
        clustering_page()
    elif page == "🎼 Music Recommendation":
        recommendation_page(df, audio_features)
    elif page == "📄 Project Report":
        report_page()
    elif page == "👨‍💻 About Developer":
        about_page()

def home_page():
    st.markdown(f"""
    <div class="logo-container">
        {AMAZON_MUSIC_LOGO}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
        <h1>Amazon Music Clustering & Recommendation Dashboard</h1>
        <div style="font-size: 1.3rem; margin-bottom: 2rem; color: #37475A;">
            🎶 Discover music patterns through advanced clustering techniques 🎶
        </div>
        <div style="font-size: 1.1rem; color: #37475A;">
            ✨ Created By Malathi Y with lots of ❤️❤️❤️✨
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; font-size: 2rem;">
        <span class="music-note" style="animation-delay: 0s;">🎵</span>
        <span class="music-note" style="animation-delay: 0.5s;">🎶</span>
        <span class="music-note" style="animation-delay: 1s;">🎵</span>
        <span class="music-note" style="animation-delay: 1.5s;">🎶</span>
        <span class="music-note" style="animation-delay: 2s;">🎵</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Project Objective</h3>
            <div style="height: 3px; background: linear-gradient(90deg, #00A8E1, #0077B6); margin: 10px 0; border-radius: 2px;"></div>
            <p>Automatically group similar songs based on their audio characteristics using
            <strong>unsupervised machine learning</strong> techniques, without relying on manual genre labels.</p>
            <div style="margin-top: 1rem; color: #00A8E1;">
                🔥 <strong>Zero Manual Labeling Required!</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset Features</h3>
            <div style="height: 3px; background: linear-gradient(90deg, #FF9900, #F28C38); margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 1.8;">
                <li><strong>🕺 Danceability:</strong> Perfect for the dance floor</li>
                <li><strong>⚡ Energy:</strong> Intensity and power levels</li>
                <li><strong>😊 Valence:</strong> Musical mood and positiveness</li>
                <li><strong>🎼 Tempo:</strong> Beats per minute (BPM)</li>
                <li><strong>🎤 And 5 more features...</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2>🔬 Advanced Clustering Algorithms</h2>
        <div style="height: 4px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 1rem auto; border-radius: 2px; width: 300px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h4>K-Means Clustering</h4>
            <div style="height: 2px; background: #00A8E1; margin: 10px 0; border-radius: 1px;"></div>
            <p><strong>🔄 Approach:</strong> Partitional clustering</p>
            <p><strong>🎪 Best for:</strong> Spherical clusters</p>
            <p><strong>📈 Optimization:</strong> Elbow method + Silhouette score</p>
            <p><strong>⚡ Advantage:</strong> Fast & efficient</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,168,225,0.1); border-radius: 8px; text-align: center;">
                <strong>🏆 Most Popular Choice</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h4>DBSCAN Clustering</h4>
            <div style="height: 2px; background: #FF9900; margin: 10px 0; border-radius: 1px;"></div>
            <p><strong>🌊 Approach:</strong> Density-based clustering</p>
            <p><strong>🎨 Best for:</strong> Irregular shapes</p>
            <p><strong>⚙️ Parameters:</strong> eps + min_samples</p>
            <p><strong>🎯 Advantage:</strong> Finds outliers</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(255,153,0,0.1); border-radius: 8px; text-align: center;">
                <strong>🔥 Noise Detection</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🌳</div>
            <h4>Hierarchical Clustering</h4>
            <div style="height: 2px; background: #0077B6; margin: 10px 0; border-radius: 1px;"></div>
            <p><strong>🌲 Approach:</strong> Agglomerative clustering</p>
            <p><strong>🎄 Best for:</strong> Nested structures</p>
            <p><strong>📊 Visualization:</strong> Dendrogram trees</p>
            <p><strong>🎨 Advantage:</strong> No preset clusters</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,119,182,0.1); border-radius: 8px; text-align: center;">
                <strong>🌟 Visual Hierarchy</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2>🎼 Intelligent Music Recommendation Engine</h2>
        <div style="height: 4px; background: linear-gradient(90deg, #FF9900, #00A8E1); margin: 1rem auto; border-radius: 2px; width: 400px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🤖🎵</div>
        <h3>Content-Based Filtering System</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0;">
            <div style="padding: 1rem; background: rgba(0,168,225,0.1); border-radius: 10px;">
                <div style="font-size: 2rem;">🎯</div>
                <strong>Cosine Similarity</strong><br>
                <small>Mathematical precision in music matching</small>
            </div>
            <div style="padding: 1rem; background: rgba(255,153,0,0.1); border-radius: 10px;">
                <div style="font-size: 2rem;">⚡</div>
                <strong>Real-time Analysis</strong><br>
                <small>Instant recommendations on demand</small>
            </div>
            <div style="padding: 1rem; background: rgba(0,119,182,0.1); border-radius: 10px;">
                <div style="font-size: 2rem;">🎨</div>
                <strong>Multi-feature Matching</strong><br>
                <small>Tempo, energy, mood & more</small>
            </div>
        </div>
        <p>This system analyzes <strong>9 distinct audio characteristics</strong> to find songs that truly resonate with your musical taste.
        No more random suggestions – every recommendation is scientifically crafted! 🧬🎶</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <div class="metric-card" style="background: #FFFFFF;">
            <h3>🚀 Ready to Explore?</h3>
            <p>Dive into the world of musical data science and discover hidden patterns in your favorite songs!</p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="padding: 1rem; background: rgba(0,168,225,0.1); border-radius: 15px; min-width: 150px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
                    <strong>Start Clustering</strong>
                </div>
                <div style="padding: 1rem; background: rgba(255,153,0,0.1); border-radius: 15px; min-width: 150px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎵</div>
                    <strong>Get Recommendations</strong>
                </div>
                <div style="padding: 1rem; background: rgba(0,119,182,0.1); border-radius: 15px; min-width: 150px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <strong>View Analytics</strong>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def clustering_page():
    st.markdown('<h1>📊 Clustering Visualization</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 K-Means", "🔍 DBSCAN", "🌳 Hierarchical"])
    
    with tab1:
        st.subheader("K-Means Clustering")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Parameters:</strong> n_clusters = 4, random_state = 42</p>
            <p><strong>Silhouette Score:</strong> 0.556</p>
            <p><strong>Davies-Bouldin Index:</strong> 0.687</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Cluster Characteristics")
        fig = px.imshow(kmeans_df.T, color_continuous_scale='Blues', title="K-Means Cluster Profiles",
                        labels=dict(x="Cluster", y="Feature", color="Value"))
        fig.update_layout(font_color='#232F3E', title_font_color='#232F3E', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Distribution")
        fig_bar = go.Figure(data=[
            go.Bar(x=kmeans_df.columns, y=kmeans_df.loc['0'], name='Cluster 0', marker_color='#00A8E1'),
            go.Bar(x=kmeans_df.columns, y=kmeans_df.loc['1'], name='Cluster 1', marker_color='#FF9900'),
            go.Bar(x=kmeans_df.columns, y=kmeans_df.loc['2'], name='Cluster 2', marker_color='#0077B6'),
            go.Bar(x=kmeans_df.columns, y=kmeans_df.loc['3'], name='Cluster 3', marker_color='#F28C38')
        ])
        fig_bar.update_layout(title="K-Means Feature Distribution", xaxis_title="Features", yaxis_title="Mean Value",
                              barmode='group', font_color='#232F3E', title_font_color='#232F3E',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("DBSCAN Clustering")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Parameters:</strong> eps = 0.8, min_samples = 5</p>
            <p><strong>Number of clusters detected:</strong> 4</p>
            <p><strong>Noise points:</strong> 0</p>
            <p><strong>Silhouette Score:</strong> 0.639</p>
            <p><strong>Davies-Bouldin Index:</strong> 0.653</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Cluster Characteristics")
        fig = px.imshow(dbscan_df.T, color_continuous_scale='Blues', title="DBSCAN Cluster Profiles",
                        labels=dict(x="Cluster", y="Feature", color="Value"))
        fig.update_layout(font_color='#232F3E', title_font_color='#232F3E', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Distribution")
        fig_bar = go.Figure(data=[
            go.Bar(x=dbscan_df.columns, y=dbscan_df.loc['0'], name='Cluster 0', marker_color='#00A8E1'),
            go.Bar(x=dbscan_df.columns, y=dbscan_df.loc['1'], name='Cluster 1', marker_color='#FF9900'),
            go.Bar(x=dbscan_df.columns, y=dbscan_df.loc['2'], name='Cluster 2', marker_color='#0077B6'),
            go.Bar(x=dbscan_df.columns, y=dbscan_df.loc['3'], name='Cluster 3', marker_color='#F28C38')
        ])
        fig_bar.update_layout(title="DBSCAN Feature Distribution", xaxis_title="Features", yaxis_title="Mean Value",
                              barmode='group', font_color='#232F3E', title_font_color='#232F3E',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.subheader("Hierarchical Clustering")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Parameters:</strong> Linkage method = ward, Number of clusters = 4</p>
            <p><strong>Silhouette Score:</strong> 0.639</p>
            <p><strong>Davies-Bouldin Index:</strong> 0.540</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Cluster Characteristics")
        fig = px.imshow(hierarchical_df.T, color_continuous_scale='Blues', title="Hierarchical Cluster Profiles",
                        labels=dict(x="Cluster", y="Feature", color="Value"))
        fig.update_layout(font_color='#232F3E', title_font_color='#232F3E', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Distribution")
        fig_bar = go.Figure(data=[
            go.Bar(x=hierarchical_df.columns, y=hierarchical_df.loc['0'], name='Cluster 0', marker_color='#00A8E1'),
            go.Bar(x=hierarchical_df.columns, y=hierarchical_df.loc['1'], name='Cluster 1', marker_color='#FF9900'),
            go.Bar(x=hierarchical_df.columns, y=hierarchical_df.loc['2'], name='Cluster 2', marker_color='#0077B6'),
            go.Bar(x=hierarchical_df.columns, y=hierarchical_df.loc['3'], name='Cluster 3', marker_color='#F28C38')
        ])
        fig_bar.update_layout(title="Hierarchical Feature Distribution", xaxis_title="Features", yaxis_title="Mean Value",
                              barmode='group', font_color='#232F3E', title_font_color='#232F3E',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

def recommendation_page(df, audio_features):
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🎼 Amazon Music Recommendation Engine</h1>
        <div style="height: 4px; background: linear-gradient(90deg, #FF9900, #00A8E1); margin: 1rem auto; border-radius: 2px; width: 400px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card" style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖🎵</div>
        <p>Select a song from our curated collection and let our AI discover your perfect musical matches!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        song_options = [f"🎵 {row['track_name']} - {row['artist_name']}" for _, row in df.iterrows()]
        selected_song = st.selectbox("🎯 Choose your favorite song:", song_options)
    
    with col2:
        n_recommendations = st.slider("📊 Recommendations:", 1, 20, 5)
    
    if st.button("🚀 Get My Recommendations!", key="recommend_btn"):
        selected_index = song_options.index(selected_song)
        
        with st.spinner("🔍 Analyzing your musical taste..."):
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df[audio_features])
            similarity_matrix = cosine_similarity(features_scaled)
            sim_scores = similarity_matrix[selected_index]
            top_indices = np.argsort(sim_scores)[::-1][1:n_recommendations+1]
        
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3>Your Selected Song</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_song_data = df.iloc[selected_index]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🎵</div>
                <h4>Track Details</h4>
                <div style="height: 2px; background: #00A8E1; margin: 10px 0; border-radius: 1px;"></div>
                <p><strong>🎼 Track:</strong> {selected_song_data['track_name']}</p>
                <p><strong>🎤 Artist:</strong> {selected_song_data['artist_name']}</p>
                <p><strong>⏱️ Duration:</strong> {selected_song_data['duration_ms']*1000:.1f} minutes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            features_values = [selected_song_data[feature] for feature in audio_features[:6]]
            feature_labels = [f.title() for f in audio_features[:6]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=features_values,
                theta=feature_labels,
                fill='toself',
                name='Selected Song',
                line_color='#00A8E1',
                fillcolor='rgba(0,168,225,0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor='rgba(0,0,0,0.2)'),
                           angularaxis=dict(gridcolor='rgba(0,0,0,0.2)')),
                title="🎯 Audio Features Profile",
                font_color='#232F3E',
                title_font_color='#232F3E',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0 2rem 0;">
            <h2>🎁 Your Personalized Recommendations</h2>
            <div style="height: 3px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 1rem auto; border-radius: 2px; width: 300px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        for i, idx in enumerate(top_indices, 1):
            rec_song = df.loc[idx]
            sim_score = sim_scores[idx]
            with st.expander(f"🎵 {i}. {rec_song['track_name']} - {rec_song['artist_name']} (Similarity: {sim_score:.3f})", expanded=(i <= 3)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎼 Primary Features</h4>
                        <div style="height: 2px; background: #FF9900; margin: 10px 0; border-radius: 1px;"></div>
                    """, unsafe_allow_html=True)
                    for feature in audio_features[:5]:
                        value = rec_song[feature]
                        color = "#00A8E1" if value > 0.7 else "#FF9900" if value > 0.4 else "#0077B6"
                        st.markdown(f"<p style='color: {color};'>🎯 <strong>{feature.title()}:</strong> {value:.3f}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⚡ Additional Attributes</h4>
                        <div style="height: 2px; background: #0077B6; margin: 10px 0; border-radius: 1px;"></div>
                    """, unsafe_allow_html=True)
                    for feature in audio_features[5:]:
                        value = rec_song[feature]
                        color = "#00A8E1" if value > 0.7 else "#FF9900" if value > 0.4 else "#0077B6"
                        st.markdown(f"<p style='color: {color};'>🎵 <strong>{feature.title()}:</strong> {value:.3f}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

def report_page():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Amazon Music Clustering & Recommendation Report</h1>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔬 Clustering Analysis", "🎼 Recommendation System", "💡 Insights & Observations"])
    with tab1:
        st.markdown("""
        <div class="metric-card">
        ♥ <strong>Dataset Name:</strong> Amazon Music Clustering<br>
        ♥ <strong>Source:</strong> Amazon music platform<br>
        ♥ <strong>Data Format:</strong> CSV file<br>
        ♥ <strong>Number of Records (Rows):</strong> 95,837<br>
        ♥ <strong>Number of Features (Columns):</strong> 23<br>
        ♥ <strong>Feature Types:</strong><br>
           ♦ 7 categorical (object/string)<br>
           ♦ 12 numerical (float)<br>
           ♦ 4 numerical (integer)<br>
        ♥ <strong>Columns in Dataset:</strong><br>
        1. id_songs – Unique identifier of each song<br>
        2. name_song – Name/title of the track<br>
        3. popularity_songs – Popularity score of the song (0–100)<br>
        4. duration_ms – Duration of the track in milliseconds<br>
        5. explicit – Explicit content flag (0 = No, 1 = Yes)<br>
        6. id_artists – Unique identifier of the artist(s)<br>
        7. release_date – Release date of the track<br>
        8. danceability – Danceability score (0.0 to 1.0)<br>
        9. energy – Energy/intensity of the track<br>
        10. key – Musical key of the song (0–11)<br>
        11. loudness – Overall loudness in decibels (dB)<br>
        12. mode – Modality of the song (major = 1, minor = 0)<br>
        13. speechiness – Presence of spoken words in the track<br>
        14. acousticness – Acoustic quality score<br>
        15. instrumentalness – Likelihood of being instrumental<br>
        16. liveness – Probability of live performance<br>
        17. valence – Musical positiveness/mood score<br>
        18. tempo – Tempo of the track in BPM (beats per minute)<br>
        19. time_signature – Time signature (beats per bar)<br>
        20. followers – Number of followers of the artist(s)<br>
        21. genres – Genre(s) associated with the artist(s)<br>
        22. name_artists – Name(s) of the artist(s)<br>
        23. popularity_artists – Popularity score of the artist(s)<br>
        ♥ <strong>Timeframe Covered:</strong> Songs range from early 20th-century releases (1920s) to modern tracks (up to 2020).<br>
        ♥ <strong>Brief Description:</strong><br>
        This dataset captures a wide spectrum of audio features, popularity metrics, and artist metadata for Amazon Music/Spotify-like tracks. It combines song-level attributes (e.g., duration, danceability, energy, valence, tempo, loudness, mode), artist-level characteristics (followers, genres, popularity), and contextual data (release date, explicit flag). It is designed to support exploratory analysis, clustering, and recommendation modeling, making it ideal for studying music similarity, genre trends, artist influence on popularity, and playlist generation.<br>
        <strong>Clustering techniques implemented:</strong><br>
        - K-Means Clustering (partitional, optimal k = 4 via silhouette score)<br>
        - DBSCAN (density-based, eps=0.8, min_samples=5)<br>
        - Hierarchical Clustering (agglomerative, linkage=ward, n_clusters=4)
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        st.header("2.1 K-Means Clustering")
        st.markdown("""
        <div class="metric-card">
        <strong>Parameters:</strong> n_clusters = 4, random_state = 42<br>
        <strong>Silhouette Score:</strong> 0.556<br>
        <strong>Davies-Bouldin Index:</strong> 0.687
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Cluster Characteristics (mean values per cluster)")
        st.dataframe(kmeans_df.round(3), use_container_width=True)
        
        st.header("2.2 DBSCAN Clustering")
        st.markdown("""
        <div class="metric-card">
        <strong>Parameters:</strong> eps = 0.8, min_samples = 5<br>
        <strong>Number of clusters detected:</strong> 4<br>
        <strong>Noise points:</strong> 0<br>
        <strong>Silhouette Score:</strong> 0.639<br>
        <strong>Davies-Bouldin Index:</strong> 0.653
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Cluster Characteristics (mean values per cluster)")
        st.dataframe(dbscan_df.round(3), use_container_width=True)
        
        st.header("2.3 Hierarchical Clustering")
        st.markdown("""
        <div class="metric-card">
        <strong>Parameters:</strong> Linkage method = ward, Number of clusters = 4<br>
        <strong>Silhouette Score:</strong> 0.639<br>
        <strong>Davies-Bouldin Index:</strong> 0.540
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Cluster Characteristics (mean values per cluster)")
        st.dataframe(hierarchical_df.round(3), use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="metric-card">
        Content-based approach using cosine similarity on audio features.<br>
        <strong>Key features:</strong> danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo<br>
        <strong>Example:</strong> For the original song (index 0), recommended songs from same cluster include 3 tracks with similar audio feature profiles.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="metric-card">
        - High-energy, high-valence clusters correlate with danceable tracks.<br>
        - High acousticness and instrumentalness dominate calm or experimental clusters.<br>
        - DBSCAN identifies outlier tracks effectively (though in this dataset noise=0).<br>
        - Hierarchical clustering reveals nested relationships between clusters.<br>
        - Cluster-based recommendations enhance playlist diversity.<br>
        <strong>Business Implications:</strong><br>
        - Cluster-based playlists improve user engagement.<br>
        - Outliers highlight unique tracks for niche audiences.<br>
        - Cluster insights support targeted marketing and mood-based playlists.
        </div>
        """, unsafe_allow_html=True)

def about_page():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>👩‍💻 About The Developer</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image("malathi.png", width=400)
            st.markdown("<div style='text-align: center; font-size: 2.1rem; font-weight: 900; margin-top: 0.7em;'>Malathi Y</div>", unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style='width: 220px; height: 220px; background: linear-gradient(135deg, #00A8E1, #FF9900);
                        border-radius: 50%; display: flex; align-items: center; justify-content: center;
                        color: white; font-size: 3rem; margin: 0 auto;'>
                👩‍💻
            </div>
            <p style='text-align: center; margin-top: 10px; font-weight: bold;'>Malathi Y</p>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>👋 Hello! I'm Malathi Y</h3>
            <div style="height: 3px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 10px 0; border-radius: 2px;"></div>
            <p>I'm a former <span style="color: #00A8E1; font-weight: bold;">Staff Nurse</span> from India (Tamil Nadu), now transitioning into <span style="color: #FF9900; font-weight: bold;">Data Science & Machine Learning</span>.<br>
            My journey from healthcare to analytics is driven by curiosity, a love for problem-solving, and deep interest in how <span style="font-weight: bold;">data</span> shapes decision-making.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    bg_cols = st.columns(3)
    
    with bg_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <h3>👩‍⚕️ My Past Profession</h3>
            <div style="height: 3px; background: #00A8E1; margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 1.8;">
                <li>🏥 Former <strong>Registered Staff Nurse</strong></li>
                <li>👩‍💼 Clinical decision-making expert</li>
                <li>💡 Healthcare data analytics enthusiast</li>
                <li>🔄 Love to take care of patients</li>
                <li>🔄 Transitioning to Data Science</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with bg_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 My Present Mission</h3>
            <div style="height: 3px; background: #FF9900; margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 1.8;">
                <li>🔄 Career Shift and kept baby step in to <strong>Data Science</strong></li>
                <li>🎯 Currently enrolled at <strong>GUVI</strong> for data science course</li>
                <li>🧠 Learning ML, AI, and Analytics</li>
                <li>📊 Building real-world projects</li>
                <li>🤝 Open to collaboration opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with bg_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <h3>🛠️ My Skills So Far</h3>
            <div style="height: 3px; background: #0077B6; margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 1.8;">
                <li>🐍 Python, SQL, Pandas, NumPy</li>
                <li>📊 Statistics & Probability</li>
                <li>🧩 Data cleaning, EDA, Data preprocessing</li>
                <li>🤖 Machine Learning (Scikit-learn)</li>
                <li>📊 Streamlit, Plotly, Seaborn, Matplotlib</li>
                <li>🧩 Power BI Dashboard creating</li>
                <li>💼 Business Insight Reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🛠️</div>
            <h3>Technical Tools</h3>
            <div style="height: 3px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 2;">
                <li>🐍 <strong>Python:</strong> Core programming powerhouse</li>
                <li>🚀 <strong>Streamlit:</strong> Interactive web framework</li>
                <li>🤖 <strong>Scikit-learn:</strong> ML algorithm library</li>
                <li>📊 <strong>Pandas & NumPy:</strong> Data manipulation</li>
                <li>📈 <strong>Plotly:</strong> Dynamic visualizations</li>
                <li>🎨 <strong>Matplotlib & Seaborn:</strong> Statistical plots</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🌟</div>
            <h3>Key Innovations</h3>
            <div style="height: 3px; background: linear-gradient(90deg, #FF9900, #0077B6); margin: 10px 0; border-radius: 2px;"></div>
            <ul style="line-height: 2;">
                <li>🔄 Multi-algorithm clustering comparison</li>
                <li>⚡ Real-time parameter optimization</li>
                <li>🎯 PCA-powered cluster visualization</li>
                <li>🎵 Recommendation engine</li>
                <li>🎨 Responsive Amazon Music UI theme</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2>🎯 Project Milestones Achieved</h2>
        <div style="height: 4px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 1rem auto; border-radius: 2px; width: 350px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    achievements = [
        ("🔍", "Data Exploration & Cleaning", "Comprehensive audio feature analysis with advanced preprocessing pipelines"),
        ("🤖", "Multi-Algorithm Implementation", "K-Means, DBSCAN, and Hierarchical clustering with automated optimization"),
        ("📈", "Model Performance Tuning", "Elbow method and silhouette score analysis for optimal parameter selection"),
        ("🎨", "Interactive Visualizations", "PCA projections and dynamic cluster characteristic heatmaps"),
        ("🎵", "Recommendation Engine", "Content-based filtering using cosine similarity and feature matching"),
        ("💎", "Professional UI Design", "Amazon Music themed interface with custom animations and typography")
    ]
    
    for i in range(0, len(achievements), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(achievements):
                icon, title, desc = achievements[i]
                st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                    <h4>✅ {title}</h4>
                    <div style="height: 2px; background: #00A8E1; margin: 10px 0; border-radius: 1px;"></div>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if i + 1 < len(achievements):
                icon, title, desc = achievements[i + 1]
                color = "#FF9900" if (i + 1) % 3 == 1 else "#0077B6"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                    <h4 style="color: {color};">✅ {title}</h4>
                    <div style="height: 2px; background: {color}; margin: 10px 0; border-radius: 1px;"></div>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2>🚀 Future Enhancement Roadmap</h2>
        <div style="height: 4px; background: linear-gradient(90deg, #FF9900, #00A8E1); margin: 1rem auto; border-radius: 2px; width: 400px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 2rem 0;">
            <div style="padding: 1.5rem; background: rgba(0,168,225,0.1); border-radius: 15px; border: 1px solid rgba(0,168,225,0.3);">
                <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">🧠</div>
                <h4>Deep Learning Integration</h4>
                <p>Deep network embeddings for more sophisticated audio feature extraction and pattern recognition.</p>
            </div>
            <div style="padding: 1.5rem; background: rgba(255,153,0,0.1); border-radius: 15px; border: 1px solid rgba(255,153,0,0.3);">
                <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">📡</div>
                <h4>Real-time Streaming</h4>
                <p>Live music analysis and recommendation updates as users listen to new tracks.</p>
            </div>
            <div style="padding: 1.5rem; background: rgba(0,119,182,0.1); border-radius: 15px; border: 1px solid rgba(0,119,182,0.3);">
                <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">👥</div>
                <h4>Collaborative Filtering</h4>
                <p>User behavior analysis and social recommendation features for enhanced personalization.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎵</div>
            <h3>Ready to Rock the Music Data World?</h3>
            <p>This project demonstrates the incredible potential of AI in understanding and organizing music.
            Every song tells a story through its data! 🎼✨</p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
                <div style="padding: 1rem 2rem; background: rgba(0,168,225,0.3); border-radius: 25px; border: 2px solid #00A8E1;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📧</div>
                    <strong>Get in Touch (malathisathish2228@gmail.com)</strong>
                </div>
                <div style="padding: 1rem 2rem; background: rgba(255,153,0,0.3); border-radius: 25px; border: 2px solid #FF9900;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💼</div>
                    <strong>View Portfolio on github(https://github.com/malathisathish)</strong>
                </div>
                <div style="padding: 1rem 2rem; background: rgba(0,119,182,0.3); border-radius: 25px; border: 2px solid #0077B6;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔗</div>
                    <strong>Connect on LinkedIn ("linkedin.com/in/malathi-y-datascience",)</strong>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>📊 Project Statistics</h3>
        <div style="display: flex; justify-content: space-around; margin: 2rem 0; flex-wrap: wrap;">
            <div class="metric-card" style="min-width: 150px; margin: 0.5rem;">
                <div style="font-size: 2rem; color: #00A8E1;">500+</div>
                <div>Lines of Code</div>
            </div>
            <div class="metric-card" style="min-width: 150px; margin: 0.5rem;">
                <div style="font-size: 2rem; color: #FF9900;">3</div>
                <div>ML Algorithms</div>
            </div>
            <div class="metric-card" style="min-width: 150px; margin: 0.5rem;">
                <div style="font-size: 2rem; color: #0077B6;">9</div>
                <div>Audio Features</div>
            </div>
      </div>
    """, unsafe_allow_html=True)
    
    # Inspirational Quote
    st.markdown("""
<div style='text-align: center; padding: 28px; 
            background: linear-gradient(135deg, #f0e6ff, #ffe8d6, #fff4e1); 
            border-radius: 20px; border: 2px dashed #a16eff; margin: 30px 0;
            box-shadow: 0 6px 20px rgba(161, 110, 255, 0.2);'>
    <blockquote style='font-size: 1.5rem; font-style: italic; color: #7a43b6; 
                      text-shadow: 1px 1px 3px rgba(0,0,0,0.1); line-height: 1.65; margin-bottom: 15px;'>
        <strong>"From saving lives in the ICU to predicting revenue through ML, 
        I'm on a mission to make data-driven insights count."</strong> 🙏
    </blockquote>
    <cite style='color: #d5489c; font-size: 1.15rem; font-weight: bold;'>
        - Malathi Y, Data Science Enthusiast
        ☻ With ❤️ from Tamilnadu, India
    </cite>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()