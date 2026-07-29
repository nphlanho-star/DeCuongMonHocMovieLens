import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import requests
import os
import sys
import subprocess
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Hỗ trợ mô hình AI Ngữ nghĩa (Sentence Transformers)
try:
    from sentence_transformers import SentenceTransformer

    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Hỗ trợ render Đồ thị tri thức tương tác
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components

    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# ==============================================================================
# CẤU HÌNH TỰ ĐỘNG KÍCH HOẠT HẠ TẦNG STREAMLIT
# ==============================================================================
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    if os.environ.get("IS_STREAMLIT_RUNNING") != "1":
        os.environ["IS_STREAMLIT_RUNNING"] = "1"
        print("🚀 Đang kích hoạt hạ tầng tính toán phân tán cao cấp...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit()

# ==============================================================================
# QUẢN LÝ PHẦN CỨNG VẬT LÝ
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# GIAO DIỆN CINEMATIC PREMIUM
# ==============================================================================
st.set_page_config(page_title="Netflix Enterprise v6.5 AI Master", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050508; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #888888; background-color: transparent; font-weight: bold; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #E50914 !important; border-bottom-color: #E50914 !important; }

    div[data-testid="stBlock"] { 
        background: linear-gradient(135deg, #0e0e12, #15151f); 
        padding: 16px; border-radius: 14px; border: 1px solid #1f1f2e; 
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1); margin-bottom: 25px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.7); 
    }
    div[data-testid="stBlock"]:hover { 
        transform: translateY(-8px) scale(1.02); border-color: #E50914; 
        box-shadow: 0 15px 35px rgba(229,9,20,0.3); 
    }
    .genre-tag { display: inline-block; background-color: #171724; color: #ff3f34; padding: 2px 10px; border-radius: 20px; font-size: 10px; margin-right: 4px; margin-bottom: 4px; border: 1px solid #2b2b3d; font-weight: bold; }
    .metric-badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .badge-success { background-color: rgba(46, 213, 115, 0.15); color: #2ed573; }
    .badge-warning { background-color: rgba(255, 159, 67, 0.15); color: #ff9f43; }
    .badge-xai { background-color: rgba(0, 210, 211, 0.15); color: #00d2d3; font-size: 10px; padding: 2px 5px; border-radius: 3px; }
    .stChatMessage { background-color: transparent !important; border: 1px solid #1f1f2e; border-radius: 10px; }
    .overview-text { font-size: 11px; color: #aaaaaa; height: 48px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# KIẾN TRÚC MẠNG NEURAL NCF
# ==============================================================================
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users=85307, num_movies=10524, embedding_dim=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, user_indices, movie_indices):
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        x = torch.cat([user_embed, movie_embed], dim=-1)
        return self.fc_layers(x).squeeze()


def predict_with_custom_user_embed(model, custom_user_embed, movie_indices_tensor):
    model.eval()
    movie_embed = model.movie_embedding(movie_indices_tensor)
    user_embed_expanded = custom_user_embed.unsqueeze(0).expand(movie_embed.size(0), -1)
    x = torch.cat([user_embed_expanded, movie_embed], dim=-1)
    return model.fc_layers(x).squeeze()


# ==============================================================================
# HÀM LẤY ẢNH POSTER VỚI BẢO VỆ DỰ PHÒNG & CACHE
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_movie_poster(movie_title, poster_url_from_df=None):
    fallback_image = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"

    if poster_url_from_df and isinstance(poster_url_from_df, str) and poster_url_from_df.startswith(
            "http") and poster_url_from_df != fallback_image:
        return poster_url_from_df

    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    clean_title = re.sub(r'\s*\(\d{4}\)', '', str(movie_title)).strip()
    if ", The" in clean_title:
        clean_title = "The " + clean_title.replace(", The", "").strip()
    elif ", A" in clean_title:
        clean_title = "A " + clean_title.replace(", A", "").strip()
    elif ", An" in clean_title:
        clean_title = "An " + clean_title.replace(", An", "").strip()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={requests.utils.quote(clean_title)}"

    try:
        response = requests.get(url, headers=headers, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            if data and 'results' in data and len(data['results']) > 0:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    return fallback_image


# ==============================================================================
# QUẢN LÝ TÀI NGUYÊN & ENGINE NHÚNG AI
# ==============================================================================
@st.cache_resource
def load_resources():
    # 1. Load Model NCF
    model = NeuralCollaborativeFiltering()
    if os.path.exists('ncf_movie_recommender.pth'):
        model.load_state_dict(torch.load('ncf_movie_recommender.pth', map_location=torch.device('cpu')))
    model = model.to(device)

    # 2. Đọc file dữ liệu
    csv_path = 'movies_enriched.csv' if os.path.exists('movies_enriched.csv') else 'movies_mapped.csv'
    movies_df = pd.read_csv(csv_path)

    if 'overview' not in movies_df.columns: movies_df['overview'] = "No overview available."
    if 'poster_url' not in movies_df.columns: movies_df['poster_url'] = ""

    # 3. Khởi tạo TF-IDF cho cốt truyện
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    enhanced_corpus = (
            movies_df['genres'].str.replace('|', ' ', regex=False) + ' ' +
            movies_df['title'] + ' ' +
            movies_df['overview'].fillna('')
    )
    tfidf_matrix = tfidf.fit_transform(enhanced_corpus)

    # 4. Khởi tạo mô hình Semantic Transformers (SBERT)
    sbert_model = None
    overview_embeddings = None
    if HAS_SBERT:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_texts = (movies_df['title'] + ". " + movies_df['overview'].fillna('')).tolist()
        overview_embeddings = sbert_model.encode(corpus_texts, show_progress_bar=False, convert_to_numpy=True)

    return model, movies_df, tfidf_matrix, tfidf, sbert_model, overview_embeddings


model, movies_df, tfidf_matrix, tfidf_vectorizer, sbert_model, overview_embeddings = load_resources()
all_genres = sorted(list(set([g for g_list in movies_df['genres'].dropna().str.split('|') for g in g_list])))

# ==============================================================================
# DASHBOARD CONTROL PANEL
# ==============================================================================
st.title("🎬 NETFLIX ENTERPRISE SYSTEMS - V6.5 AI MASTER")
st.caption("Kiến trúc công nghiệp: Hybrid Sentence-BERT + TF-IDF Keyword Boosting | Neural NCF | Knowledge Graph")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1: st.metric(label="🖥️ Thiết bị tính toán", value=str(device).upper())
with m_col2: st.metric(label="📊 Chiều Vector NCF", value="32-Dimensions")
with m_col3: st.metric(label="🧠 Động cơ Semantic", value="Hybrid SBERT Active" if HAS_SBERT else "TF-IDF Mode")
with m_col4: st.metric(label="🔬 Đồ thị không gian", value="PyVis Enabled" if HAS_PYVIS else "Disabled")

st.markdown("---")

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# ==============================================================================
# BỘ ĐIỀU PHỐI TẠI THANH SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("⚙️ CHẾ ĐỘ ENGINE GỢI Ý")
    app_mode = st.sidebar.radio(
        "Lựa chọn phân hệ chức năng:",
        [
            "🤖 Tìm Phim Qua Mô Tả Cốt Truyện (Semantic AI)",
            "🧪 Pha Chế Điện Ảnh (Cinematic Alchemist)",
            "👤 Cá nhân hóa AI (NCF)",
            "🔍 Tìm Phim Qua Từ Khóa/Tên Phim",
            "🌌 Đồ thị Tri thức (Knowledge Graph)",
            "🏷️ Tìm Phim Theo Thể Loại"
        ]
    )
    st.markdown("---")
    top_k = st.slider("🍿 Số lượng phim hiển thị (Top K):", min_value=5, max_value=20, value=10)

    st.markdown("---")
    st.subheader("📂 DỮ LIỆU SỬ DỤNG")
    if os.path.exists('movies_enriched.csv'):
        st.success("✅ Đang dùng `movies_enriched.csv` (Đã có Cốt truyện & Poster)")
    else:
        st.warning("⚠️ Đang dùng `movies_mapped.csv` gốc.")

# ==============================================================================
# PHÂN HỆ: TÌM PHIM BẰNG MÔ TẢ CỐT TRUYỆN
# ==============================================================================
if app_mode == "🤖 Tìm Phim Qua Mô Tả Cốt Truyện (Semantic AI)":
    st.header("🤖 Pure Plot Semantic Search (Tìm Phim Qua Mô Tả Cốt Truyện Thuần Túy)")
    st.markdown(
        "💡 **Kiến trúc tối ưu:** Ưu tiên **80% Vector Ngữ Nghĩa SBERT** cho cốt truyện, hỗ trợ **15% TF-IDF** và **5% Keyword Boost**.")

    plot_query = st.text_area(
        "📝 Nhập câu mô tả nội dung, bối cảnh, nhân vật hoặc diễn biến phim bạn muốn tìm:",
        placeholder="Ví dụ: A young wizard boy attends a magical school, makes friends, and faces dark forces...",
        height=110
    )

    if st.button("🔍 GIẢI MÃ CỐT TRUYỆN & TÌM PHIM"):
        if not plot_query.strip():
            st.error("Vui lòng nhập nội dung mô tả bộ phim!")
        else:
            with st.spinner("🧠 AI đang phân tích ma trận ngữ nghĩa cốt truyện..."):
                start_perf = time.perf_counter()

                if HAS_SBERT and overview_embeddings is not None:
                    query_vec = sbert_model.encode([plot_query], convert_to_numpy=True)
                    dense_scores = cosine_similarity(query_vec, overview_embeddings).flatten()
                else:
                    dense_scores = np.zeros(len(movies_df))

                tfidf_query = tfidf_vectorizer.transform([plot_query])
                sparse_scores = cosine_similarity(tfidf_query, tfidf_matrix).flatten()

                words = re.findall(r'\b[a-zA-Z]{3,}\b', plot_query.lower())
                stopwords = {'this', 'film', 'series', 'set', 'and', 'follows', 'the', 'for', 'with', 'one', 'most',
                             'all', 'time', 'about', 'from', 'that', 'they', 'them', 'their', 'movie', 'story'}
                keywords = [w for w in words if w not in stopwords]

                boost_scores = np.zeros(len(movies_df))
                if keywords:
                    for kw in keywords:
                        title_match = movies_df['title'].str.lower().str.contains(kw, regex=False).fillna(False)
                        overview_match = movies_df['overview'].str.lower().str.contains(kw, regex=False).fillna(False)
                        boost_scores += (title_match.astype(float) * 0.1) + (overview_match.astype(float) * 0.05)
                    max_b = np.max(boost_scores)
                    if max_b > 0:
                        boost_scores = boost_scores / max_b

                if HAS_SBERT and overview_embeddings is not None:
                    final_scores = (0.80 * dense_scores) + (0.15 * sparse_scores) + (0.05 * boost_scores)
                else:
                    final_scores = (0.80 * sparse_scores) + (0.20 * boost_scores)

                top_indices = np.argsort(final_scores)[::-1][:top_k]
                end_perf = time.perf_counter()

                st.success(
                    f"🎉 Hoàn tất phân tích ngữ nghĩa cốt truyện trong **{(end_perf - start_perf) * 1000:.2f} ms**!")

                cols = st.columns(5)
                for i, idx in enumerate(top_indices):
                    m_info = movies_df.iloc[idx]
                    title = m_info['title']
                    genres_raw = str(m_info['genres']).split('|')
                    overview_text = str(m_info.get('overview', 'No overview available.'))
                    poster_url = get_movie_poster(title, m_info.get('poster_url'))
                    match_score = min(99.9, max(50.0, final_scores[idx] * 100 if final_scores[idx] < 1.0 else 95.0))

                    with cols[i % 5]:
                        st.image(poster_url, use_container_width=True)
                        badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                        st.markdown(f"""
                            <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                            <div>{badges}</div>
                            <span style="font-size:12px; color:#2ed573; font-weight:bold;">🎯 ĐỘ KHỚP HYBRID: {match_score:.1f}%</span>
                            <div class="overview-text" title="{overview_text}">📖 {overview_text}</div>
                        """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ: PHA CHẾ ĐIỆN ẢNH AI (CINEMATIC ALCHEMIST)
# ==============================================================================
elif app_mode == "🧪 Pha Chế Điện Ảnh (Cinematic Alchemist)":
    st.header("🧪 Cinematic Alchemist Engine (Thuật Toán Pha Chế Điện Ảnh)")
    st.markdown("💡 **Cơ sở khoa học:** Trích xuất Vector Nhúng 32 chiều và thực hiện phép toán nội suy tuyến tính.")

    input_method = st.radio("Lựa chọn phương thức nhập công thức:", ["🎛️ Bảng điều khiển thanh trượt (Sliders)",
                                                                     "✍️ Nhập công thức tự nhiên (Natural Language Formula)"])

    movie_a_title = movies_df['title'].values[0]
    movie_b_title = movies_df['title'].values[1]
    ratio_a = 50

    if input_method == "🎛️ Bảng điều khiển thanh trượt (Sliders)":
        col_blend1, col_blend2 = st.columns(2)
        with col_blend1:
            movie_a_title = st.selectbox("🎬 Bộ phim gốc A:", movies_df['title'].values, index=0)
        with col_blend2:
            movie_b_title = st.selectbox("🎭 Bộ phim gốc B:", movies_df['title'].values,
                                         index=min(1, len(movies_df) - 1))

        ratio_a = st.slider("⚖️ Tỷ lệ pha trộn (Phim A đóng góp):", min_value=10, max_value=90, value=50, step=5,
                            format="%d%%")
        st.caption(
            f"🧪 **Công thức thiết lập:** `{ratio_a}%` **{movie_a_title}** + `{100 - ratio_a}%` **{movie_b_title}**")
    else:
        user_formula = st.text_input("Gõ câu lệnh pha chế của bạn tại đây:",
                                     placeholder="Ví dụ: 70% Inception (2010) + 30% Toy Story (1995)")
        if user_formula:
            percentages = re.findall(r'(\d+)\s*%', user_formula)
            parts = user_formula.split('+') if '+' in user_formula else [user_formula]

            if len(parts) >= 1:
                q_vec_a = tfidf_vectorizer.transform([parts[0]])
                sim_a = cosine_similarity(q_vec_a, tfidf_matrix).flatten()
                if np.max(sim_a) > 0.02:
                    movie_a_title = movies_df.iloc[np.argsort(sim_a)[::-1][0]]['title']
                if len(percentages) >= 1: ratio_a = int(percentages[0])

            if len(parts) >= 2:
                q_vec_b = tfidf_vectorizer.transform([parts[1]])
                sim_b = cosine_similarity(q_vec_b, tfidf_matrix).flatten()
                if np.max(sim_b) > 0.02:
                    movie_b_title = movies_df.iloc[np.argsort(sim_b)[::-1][0]]['title']

            st.info(
                f"🔮 **AI nhận diện:** Trộn `{ratio_a}%` **[{movie_a_title}]** + `{100 - ratio_a}%` **[{movie_b_title}]**")

    if st.button("🧪 TIẾN HÀNH PHẢN ỨNG VÀ HÒA TRỘN VECTOR"):
        if movie_a_title == movie_b_title:
            st.error("Lỗi công thức: Vui lòng lựa chọn 2 bộ phim khác biệt nhau!")
        else:
            with st.spinner("🔬 Đang kích hoạt lò phản ứng tuyến tính..."):
                start_perf = time.perf_counter()

                movie_a_info = movies_df[movies_df['title'] == movie_a_title].iloc[0]
                movie_b_info = movies_df[movies_df['title'] == movie_b_title].iloc[0]

                idx_a = int(movie_a_info.get('movie_idx', movie_a_info.name))
                idx_b = int(movie_b_info.get('movie_idx', movie_b_info.name))

                model.eval()
                with torch.no_grad():
                    movie_embeddings = model.movie_embedding.weight.detach().cpu().numpy()

                vec_a = movie_embeddings[idx_a]
                vec_b = movie_embeddings[idx_b]

                w_a = ratio_a / 100.0
                w_b = 1.0 - w_a
                vec_blended = (w_a * vec_a) + (w_b * vec_b)
                vec_blended = vec_blended.reshape(1, -1)

                blended_sims = cosine_similarity(vec_blended, movie_embeddings).flatten()
                sorted_idxs = np.argsort(blended_sims)[::-1]

                valid_recs = []
                for idx in sorted_idxs:
                    m_title = movies_df.iloc[idx]['title']
                    if m_title != movie_a_title and m_title != movie_b_title:
                        valid_recs.append(idx)
                    if len(valid_recs) == top_k: break

                end_perf = time.perf_counter()
                st.success(f"🎉 Hòa trộn hoàn tất trong **{(end_perf - start_perf) * 1000:.2f} ms**!")

                cols = st.columns(5)
                for i, idx in enumerate(valid_recs):
                    m_info = movies_df.iloc[idx]
                    title = m_info['title']
                    genres_raw = str(m_info['genres']).split('|')
                    poster_url = get_movie_poster(title, m_info.get('poster_url'))
                    match_percent = blended_sims[idx] * 100

                    with cols[i % 5]:
                        st.image(poster_url, use_container_width=True)
                        badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                        st.markdown(f"""
                            <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                            <div>{badges}</div>
                            <span style="font-size:12px; color:#E50914; font-weight:bold;">🧪 ĐỘ HÒA HỢP: {match_percent:.1f}%</span>
                        """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ: CÁ NHÂN HÓA AI (NCF)
# ==============================================================================
elif app_mode == "👤 Cá nhân hóa AI (NCF)":
    st.header("🎯 Gợi ý cá nhân hóa Deep Learning (Bayesian NCF)")
    cold_start_mode = st.checkbox("🆕 Tôi là người dùng mới (Giả lập Khởi động lạnh - Cold Start)")
    user_id_input = 80000
    selected_genre = "Tất cả thể loại"
    cold_start_movies = []

    if not cold_start_mode:
        col_input1, col_input2 = st.columns([2, 2])
        with col_input1:
            user_id_input = st.number_input("Nhập mã định danh User ID (0 - 85306):", min_value=0, max_value=85306,
                                            value=80000)
        with col_input2:
            selected_genre = st.selectbox("Lọc nhanh theo dòng phim:", ["Tất cả thể loại"] + all_genres)
    else:
        cold_start_movies = st.multiselect("Chọn tối thiểu 3 bộ phim bạn yêu thích nhất:", movies_df['title'].values)
        selected_genre = st.selectbox("Lọc nhanh theo dòng phim:", ["Tất cả thể loại"] + all_genres)

    if st.button("🚀 TRUY XUẤT HỒ SƠ VECTOR"):
        if cold_start_mode and len(cold_start_movies) < 3:
            st.error("Vui lòng chọn ít nhất 3 bộ phim để phân tích!")
        else:
            with st.spinner('Hạ tầng đang phân tích Vector...'):
                start_perf = time.perf_counter()
                model.train()
                movie_tensor = torch.tensor(list(range(len(movies_df))), dtype=torch.long).to(device)

                if cold_start_mode:
                    selected_indices = [movies_df[movies_df['title'] == title].index[0] for title in cold_start_movies]
                    selected_indices_t = torch.tensor(selected_indices, dtype=torch.long).to(device)
                    with torch.no_grad():
                        movie_embs = model.movie_embedding(selected_indices_t)
                        synthetic_user_emb = torch.mean(movie_embs, dim=0)
                else:
                    with torch.no_grad():
                        user_t_single = torch.tensor([user_id_input], dtype=torch.long).to(device)
                        synthetic_user_emb = model.user_embedding(user_t_single).squeeze(0)

                mc_samples = []
                for _ in range(5):
                    with torch.no_grad():
                        sample_preds = predict_with_custom_user_embed(model, synthetic_user_emb,
                                                                      movie_tensor).cpu().numpy()
                        mc_samples.append(sample_preds)

                mc_samples = np.array(mc_samples)
                mean_scores = np.mean(mc_samples, axis=0)
                std_scores = np.std(mc_samples, axis=0)
                all_indices = np.argsort(mean_scores)[::-1]
                valid_indices, valid_scores, valid_std, valid_alignments = [], [], [], []

                with torch.no_grad():
                    all_movie_embs = model.movie_embedding(movie_tensor)
                    u_norm = synthetic_user_emb / synthetic_user_emb.norm(dim=-1, keepdim=True)
                    m_norm = all_movie_embs / all_movie_embs.norm(dim=-1, keepdim=True)
                    alignments = torch.matmul(m_norm, u_norm).cpu().numpy()

                for idx in all_indices:
                    movie_info = movies_df.iloc[idx]
                    if selected_genre == "Tất cả thể loại" or selected_genre in str(movie_info['genres']).split('|'):
                        valid_indices.append(idx)
                        valid_scores.append(mean_scores[idx])
                        valid_std.append(std_scores[idx])
                        valid_alignments.append(alignments[idx])
                    if len(valid_indices) == top_k: break

                end_perf = time.perf_counter()

                if not valid_indices:
                    st.warning("Không tìm thấy bộ phim nào phù hợp.")
                else:
                    cols = st.columns(5)
                    for i, idx in enumerate(valid_indices):
                        movie_info = movies_df.iloc[idx]
                        title = movie_info['title']
                        genres_raw = str(movie_info['genres']).split('|')
                        poster_url = get_movie_poster(title, movie_info.get('poster_url'))
                        uncertainty = valid_std[i]

                        confidence_badge = f'<span class="metric-badge badge-success">Tin cậy cao</span>' if uncertainty < 0.05 else f'<span class="metric-badge badge-warning">Cần khám phá</span>'
                        alignment_percent = max(0.0, float(valid_alignments[i])) * 100

                        with cols[i % 5]:
                            st.image(poster_url, use_container_width=True)
                            badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                            st.markdown(f"""
                                <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                                <div style="margin-bottom:6px;">{confidence_badge}</div>
                                <div>{badges}</div>
                                <span style="font-size:12px; color:#E50914; font-weight:bold;">🔥 PHÙ HỢP: {valid_scores[i] * 100:.1f}%</span>
                                <div style="margin-top:4px;"><span class="badge-xai">🧬 Hợp gu: {alignment_percent:.1f}%</span></div>
                            """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ: TÌM PHIM THEO TÊN / TỪ KHÓA TRỰC TIẾP (ĐÃ ĐƯỢC SỬA)
# ==============================================================================
elif app_mode == "🔍 Tìm Phim Qua Từ Khóa/Tên Phim":
    st.header("🔍 Tìm Phim Trực Tiếp Theo Tên Phim / Từ Khóa")
    st.markdown(
        "💡 **Tìm kiếm tiêu đề:** Nhập tên bộ phim hoặc từ khóa bất kỳ để tìm kiếm chính xác các bộ phim có chứa tiêu đề đó.")

    search_query = st.text_input(
        "🔎 Nhập tên phim hoặc từ khóa cần tìm:",
        placeholder="Ví dụ: Toy Story, Batman, Matrix, Star Wars, Harry Potter..."
    )

    if st.button("🔍 TÌM KIẾM PHIM"):
        if not search_query.strip():
            st.error("Vui lòng nhập tên phim hoặc từ khóa tìm kiếm!")
        else:
            with st.spinner("🔍 Đang tra cứu danh mục phim..."):
                query_clean = search_query.strip().lower()

                # 1. Khớp từ khóa trực tiếp trên tiêu đề tên phim
                matched_df = movies_df[movies_df['title'].str.lower().str.contains(query_clean, regex=False)].copy()

                # 2. Dự phòng: Nếu không tìm thấy chính xác, gợi ý các tên phim có nét tương đồng nhất (Character N-gram TF-IDF)
                if matched_df.empty:
                    st.warning("⚠️ Không tìm thấy tên phim khớp hoàn toàn. Đang tìm các tiêu đề gần giống nhất...")
                    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                    title_matrix = char_vectorizer.fit_transform(movies_df['title'].str.lower())
                    query_vec = char_vectorizer.transform([query_clean])

                    scores = cosine_similarity(query_vec, title_matrix).flatten()
                    top_indices = np.argsort(scores)[::-1][:top_k]

                    # Lọc bỏ các kết quả có độ tương đồng = 0
                    valid_top = [idx for idx in top_indices if scores[idx] > 0.05]
                    results_df = movies_df.iloc[valid_top]
                else:
                    results_df = matched_df.head(top_k)

                if results_df.empty:
                    st.error("❌ Không tìm thấy bộ phim nào phù hợp với tên đã nhập!")
                else:
                    st.success(f"🎉 Tìm thấy {len(results_df)} kết quả khớp với từ khóa!")
                    cols = st.columns(5)
                    for i, (df_idx, m_info) in enumerate(results_df.iterrows()):
                        title = m_info['title']
                        genres_raw = str(m_info['genres']).split('|')
                        overview_text = str(m_info.get('overview', 'No overview available.'))
                        poster_url = get_movie_poster(title, m_info.get('poster_url'))

                        with cols[i % 5]:
                            st.image(poster_url, use_container_width=True)
                            badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                            st.markdown(f"""
                                <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                                <div>{badges}</div>
                                <div class="overview-text" title="{overview_text}">📖 {overview_text}</div>
                            """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ: ĐỒ THỊ TRI THỨC TƯƠNG TÁC (KNOWLEDGE GRAPH)
# ==============================================================================
elif app_mode == "🌌 Đồ thị Tri thức (Knowledge Graph)":
    st.header("🌌 Neural Knowledge Graph - Vũ trụ Điện ảnh 2D/3D")

    if not HAS_PYVIS:
        st.error("⚠️ Lỗi môi trường: Hệ thống yêu cầu thư viện `pyvis`. Vui lòng chạy lệnh: `pip install pyvis`")
    else:
        root_movie = st.selectbox("🎯 Chọn bộ phim làm Tâm điểm (Root Node):", movies_df['title'].values)
        num_neighbors = st.slider("Mật độ liên kết:", 5, 30, 15)

        if st.button("🚀 KHỞI TẠO KHÔNG GIAN ĐỒ THỊ"):
            with st.spinner("Đang biên dịch ma trận đồ thị..."):
                query_idx = movies_df[movies_df['title'] == root_movie].index[0]

                model.eval()
                with torch.no_grad():
                    movie_embeddings = model.movie_embedding.weight.detach().cpu().numpy()

                query_dense = movie_embeddings[query_idx].reshape(1, -1)
                sims = cosine_similarity(query_dense, movie_embeddings).flatten()
                top_neighbor_idxs = np.argsort(sims)[::-1][1:num_neighbors + 1]

                net = Network(height="600px", width="100%", bgcolor="#050508", font_color="white")
                net.add_node(root_movie, label=root_movie, color="#E50914", size=25, title="Root Movie")

                for n_idx in top_neighbor_idxs:
                    m_row = movies_df.iloc[n_idx]
                    m_title = m_row['title']
                    sim_val = sims[n_idx]
                    net.add_node(m_title, label=m_title, color="#00d2d3", size=15, title=f"Match: {sim_val * 100:.1f}%")
                    net.add_edge(root_movie, m_title, value=float(sim_val), title=f"{sim_val * 100:.1f}% Similarity")

                    primary_genre = str(m_row['genres']).split('|')[0]
                    net.add_node(primary_genre, label=primary_genre, color="#ff9f43", shape="ellipse", size=10)
                    net.add_edge(m_title, primary_genre, color="#333344")

                graph_path = "knowledge_graph.html"
                net.save_graph(graph_path)
                with open(graph_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=620, scrolling=False)

# ==============================================================================
# PHÂN HỆ: TÌM PHIM THEO THỂ LOẠI
# ==============================================================================
elif app_mode == "🏷️ Tìm Phim Theo Thể Loại":
    st.header("🏷️ Khám Phá Điện Ảnh Theo Thể Loại")
    selected_g = st.selectbox("Chọn thể loại phim:", all_genres)
    sort_by = st.selectbox("Sắp xếp theo:", ["Ngẫu nhiên khám phá", "Tên phim (A-Z)"])

    filtered_df = movies_df[movies_df['genres'].fillna('').str.contains(selected_g, regex=False)]
    if sort_by == "Tên phim (A-Z)":
        filtered_df = filtered_df.sort_values(by="title")
    else:
        filtered_df = filtered_df.sample(frac=1, random_state=42)

    display_df = filtered_df.head(top_k)

    cols = st.columns(5)
    for i, (df_idx, m_info) in enumerate(display_df.iterrows()):
        title = m_info['title']
        genres_raw = str(m_info['genres']).split('|')
        overview_text = str(m_info.get('overview', 'No overview available.'))
        poster_url = get_movie_poster(title, m_info.get('poster_url'))

        with cols[i % 5]:
            st.image(poster_url, use_container_width=True)
            badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
            st.markdown(f"""
                <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                <div>{badges}</div>
                <div class="overview-text" title="{overview_text}">📖 {overview_text}</div>
            """, unsafe_allow_html=True)