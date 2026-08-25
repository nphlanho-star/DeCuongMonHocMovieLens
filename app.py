import os
import sys
import warnings
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# TẮT CẢNH BÁO HỆ THỐNG & CẤU HÌNH TỐI ƯU TERMINAL
# ---------------------------------------------------------
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import requests
import subprocess
import time
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# KIỂM TRA & TẢI CÁC THƯ VIỆN PHỤ TRỢ (GLOBAL CHECK)
# ---------------------------------------------------------
try:
    from deep_translator import GoogleTranslator

    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    import transformers

    transformers.logging.set_verbosity_error()
    from sentence_transformers import SentenceTransformer

    HAS_SBERT_INSTALLED = True
except ImportError:
    HAS_SBERT_INSTALLED = False

try:
    from pyvis.network import Network
    import streamlit.components.v1 as components

    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# ---------------------------------------------------------
# CẤU HÌNH TỰ ĐỘNG KÍCH HOẠT HẠ TẦNG STREAMLIT
# ---------------------------------------------------------
if __name__ == '__main__' and os.environ.get("IS_STREAMLIT_RUNNING") != "1":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    os.environ["IS_STREAMLIT_RUNNING"] = "1"
    print("🚀 Đang kích hoạt hạ tầng tính toán phân tán cao cấp...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

# ---------------------------------------------------------
# QUẢN LÝ PHẦN CỨNG VẬT LÝ
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN CINEMATIC PREMIUM STREAMLIT
# ---------------------------------------------------------
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
    .overview-text { font-size: 11px; color: #aaaaaa; height: 48px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# KIẾN TRÚC MẠNG NEURAL COLLABORATIVE FILTERING (NCF)
# ---------------------------------------------------------
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users=85307, num_movies=10524, embedding_dim=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.num_movies = num_movies
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, movie_indices):
        movie_indices = torch.clamp(movie_indices, 0, self.num_movies - 1)
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        x = torch.cat([user_embed, movie_embed], dim=-1)
        return self.fc_layers(x).squeeze(-1)


def predict_with_custom_user_embed(model, custom_user_embed, movie_indices_tensor):
    model.eval()
    movie_indices_tensor = torch.clamp(movie_indices_tensor, 0, model.num_movies - 1)
    movie_embed = model.movie_embedding(movie_indices_tensor)
    user_embed_expanded = custom_user_embed.unsqueeze(0).expand(movie_embed.size(0), -1)
    x = torch.cat([user_embed_expanded, movie_embed], dim=-1)
    return model.fc_layers(x).squeeze(-1)


# ---------------------------------------------------------
# TIỆN ÍCH CHUẨN HÓA, DỊCH THUẬT AN TOÀN & TMDB API
# ---------------------------------------------------------
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
FALLBACK_IMAGE = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"


def safe_translate_text(text):
    if not HAS_TRANSLATOR or not text or not text.strip():
        return text, False

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        if not translated or "Error 500" in translated or "Server Error" in translated or "That’s an error" in translated:
            return text, False
        return translated, True
    except Exception:
        return text, False


def normalize_title(title):
    if not isinstance(title, str): return ""
    t = re.sub(r'\s*\([^)]*\)', '', title).lower().strip()
    t = re.sub(r'[^a-z0-9]', '', t)
    return t


@st.cache_data(show_spinner=False, ttl=3600)
def clean_movie_title(movie_title):
    if not isinstance(movie_title, str):
        return ""
    clean_title = re.sub(r'\s*\([^)]*\)', '', movie_title).strip()

    for prefix in ["The", "A", "An"]:
        if clean_title.endswith(f", {prefix}"):
            clean_title = f"{prefix} " + clean_title[:-len(f", {prefix}")].strip()
            break

    return clean_title


@st.cache_data(show_spinner=False, ttl=3600)
def get_tmdb_movie_details(movie_title, poster_url_from_df=None):
    clean_title = clean_movie_title(movie_title)
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(clean_title)}"

    poster_url = poster_url_from_df if (
            poster_url_from_df and isinstance(poster_url_from_df, str) and poster_url_from_df.startswith(
        "http") and poster_url_from_df != FALLBACK_IMAGE) else FALLBACK_IMAGE
    fetched_overview = None

    try:
        response = requests.get(url, headers=headers, timeout=0.8)
        if response.status_code == 200:
            data = response.json()
            if data and 'results' in data and len(data['results']) > 0:
                first_match = data['results'][0]

                if poster_url == FALLBACK_IMAGE:
                    p_path = first_match.get('poster_path')
                    if p_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{p_path}"

                fetched_overview = first_match.get('overview')
    except Exception:
        pass

    return poster_url, fetched_overview


@st.cache_data(show_spinner=False, ttl=3600)
def get_movie_trailer_url(movie_title):
    clean_title = clean_movie_title(movie_title)
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(clean_title)}"

    try:
        res = requests.get(search_url, headers=headers, timeout=0.8)
        if res.status_code == 200:
            results = res.json().get('results', [])
            if results:
                movie_id = results[0]['id']
                videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
                v_res = requests.get(videos_url, headers=headers, timeout=0.8)
                if v_res.status_code == 200:
                    videos = v_res.json().get('results', [])
                    for video in videos:
                        if video.get('site') == 'YouTube' and video.get('type') in ['Trailer', 'Teaser']:
                            return f"https://www.youtube.com/watch?v={video.get('key')}"
    except Exception:
        pass

    return f"https://www.youtube.com/results?search_query={requests.utils.quote(clean_title + ' official trailer')}"


def resolve_single_movie(m_info):
    title = m_info['title']
    df_poster = m_info.get('poster_url')
    overview = m_info.get('overview_str', 'No overview available.')

    has_valid_poster = isinstance(df_poster, str) and df_poster.startswith("http") and df_poster != FALLBACK_IMAGE
    has_valid_overview = isinstance(overview, str) and len(
        overview.strip()) > 20 and overview != "No overview available."

    if has_valid_poster and has_valid_overview:
        return title, df_poster, overview

    poster_url, tmdb_overview = get_tmdb_movie_details(title, df_poster)
    if not has_valid_overview and tmdb_overview:
        overview = tmdb_overview
    return title, poster_url, overview


def resolve_movies_batch(recs_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(resolve_single_movie, recs_list))

    for i, (title, poster_url, overview) in enumerate(results):
        recs_list[i]['poster_url'] = poster_url
        recs_list[i]['overview_text'] = overview
    return recs_list


# ---------------------------------------------------------
# TẢI TÀI NGUYÊN HỆ THỐNG (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    csv_path = 'movies_enriched.csv' if os.path.exists('movies_enriched.csv') else 'movies_mapped.csv'
    movies_df = pd.read_csv(csv_path)

    if 'overview' not in movies_df.columns: movies_df['overview'] = "No overview available."
    if 'poster_url' not in movies_df.columns: movies_df['poster_url'] = ""

    movies_df['norm_title'] = movies_df['title'].apply(normalize_title)
    movies_df['title_lower'] = movies_df['title'].fillna('').str.lower()
    movies_df['genres_list'] = movies_df['genres'].fillna('').apply(lambda x: [g for g in str(x).split('|') if g])
    movies_df['overview_str'] = movies_df['overview'].fillna('No overview available.')

    num_movies = len(movies_df)
    model = NeuralCollaborativeFiltering(num_movies=num_movies)
    if os.path.exists('ncf_movie_recommender.pth'):
        try:
            model.load_state_dict(torch.load('ncf_movie_recommender.pth', map_location=torch.device('cpu')))
        except Exception:
            pass
    model = model.to(device)

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    enhanced_corpus = (
            movies_df['genres'].fillna('').str.replace('|', ' ', regex=False) + ' ' +
            movies_df['title'].fillna('') + ' ' +
            movies_df['overview'].fillna('')
    )
    tfidf_matrix = tfidf.fit_transform(enhanced_corpus)

    sbert_model = None
    overview_embeddings = None
    has_sbert_active = False

    if HAS_SBERT_INSTALLED:
        try:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            if os.path.exists('overview_embeddings.npy'):
                overview_embeddings = np.load('overview_embeddings.npy')
            else:
                corpus_texts = (movies_df['title'].fillna('') + ". " + movies_df['overview'].fillna('')).tolist()
                overview_embeddings = sbert_model.encode(corpus_texts, show_progress_bar=False, convert_to_numpy=True)
            has_sbert_active = True
        except Exception:
            has_sbert_active = False

    return model, movies_df, tfidf_matrix, tfidf, sbert_model, overview_embeddings, has_sbert_active


model, movies_df, tfidf_matrix, tfidf_vectorizer, sbert_model, overview_embeddings, HAS_SBERT = load_resources()
all_genres = sorted(list(set([g for g_list in movies_df['genres_list'] for g in g_list if g])))

# ---------------------------------------------------------
# KHO LƯU TRỮ SESSION STATE
# ---------------------------------------------------------
if "user_store" not in st.session_state:
    st.session_state.user_store = {}

if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = 80000

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "recommendations" not in st.session_state: st.session_state.recommendations = None
if "active_trailer" not in st.session_state: st.session_state.active_trailer = None


def get_current_user_data(uid):
    if uid not in st.session_state.user_store:
        st.session_state.user_store[uid] = {"watchlist": [], "history": []}
    return st.session_state.user_store[uid]


def get_excluded_norm_titles(user_id):
    u_data = get_current_user_data(user_id)
    raw_list = u_data["watchlist"] + u_data["history"]
    return set([normalize_title(t) for t in raw_list if t])


def get_dynamic_user_embedding(model, user_id, movies_df, user_history_and_watchlist, alpha=0.4):
    model.eval()
    with torch.no_grad():
        u_t = torch.tensor([user_id], dtype=torch.long).to(device)
        base_user_emb = model.user_embedding(u_t).squeeze(0)

        if not user_history_and_watchlist:
            return base_user_emb

        norm_interacted = set([normalize_title(t) for t in user_history_and_watchlist])
        interacted_indices = movies_df[movies_df['norm_title'].isin(norm_interacted)].index.tolist()

        if not interacted_indices:
            return base_user_emb

        m_indices_t = torch.tensor(interacted_indices, dtype=torch.long).to(device)
        m_embeds = model.movie_embedding(m_indices_t)
        history_emb = torch.mean(m_embeds, dim=0)

        dynamic_emb = (alpha * base_user_emb) + ((1.0 - alpha) * history_emb)
        return dynamic_emb


# ---------------------------------------------------------
# KHUNG RENDER CARD PHIM CHUẨN CINEMATIC
# ---------------------------------------------------------
def render_movie_card(col, title, genres_raw, overview_text, poster_url, extra_info_html="", key_prefix="card"):
    cur_uid = st.session_state.current_user_id
    u_data = get_current_user_data(cur_uid)

    with col:
        st.image(poster_url, width="stretch")
        badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw if g])

        st.markdown(
            f'<h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>',
            unsafe_allow_html=True)

        if extra_info_html:
            st.markdown(extra_info_html, unsafe_allow_html=True)

        st.markdown(f'<div>{badges}</div><div class="overview-text" title="{overview_text}">📖 {overview_text}</div>',
                    unsafe_allow_html=True)

        clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', title)
        card_id = f"{key_prefix}_{clean_key}"

        rating_key = f"rate_{card_id}"
        cur_rating = st.session_state.user_ratings.get(title, 0)
        rating = st.selectbox("⭐ Active Learning", [0, 1, 2, 3, 4, 5], index=cur_rating, key=rating_key,
                              help="Đánh giá để huấn luyện mô hình")
        if rating != cur_rating:
            st.session_state.user_ratings[title] = rating
            if rating >= 4 and title not in u_data["history"]:
                u_data["history"].append(title)

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("▶️ Trailer", key=f"tr_{card_id}"):
                if st.session_state.active_trailer == card_id:
                    st.session_state.active_trailer = None
                else:
                    st.session_state.active_trailer = card_id
                    if title not in u_data["history"]:
                        u_data["history"].append(title)
                st.rerun()

        with btn_c2:
            is_saved = title in u_data["watchlist"]
            heart_icon = "❤️ Saved" if is_saved else "🤍 Save"
            if st.button(heart_icon, key=f"wl_{card_id}"):
                if is_saved:
                    u_data["watchlist"].remove(title)
                else:
                    u_data["watchlist"].append(title)
                st.rerun()

        if st.session_state.active_trailer == card_id:
            trailer_url = get_movie_trailer_url(title)
            st.markdown("---")
            if "youtube.com/watch" in trailer_url:
                st.video(trailer_url)
            else:
                st.markdown(f"[🔗 Mở trên YouTube]({trailer_url})")


# ---------------------------------------------------------
# THANH ĐIỀU HƯỚNG CHÍNH & PHÂN QUYỀN SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://assets.nflxext.com/ffe/siteui/common/icons/netflix_logo_2020.svg", width=160)
    st.markdown("### 🔑 CHỌN VAI TRÒ TRUY CẬP")
    user_role = st.selectbox("Vai trò hệ thống:", ["👤 Người Dùng (User)", "⚙️ Quản Trị Viên (Admin)"])

    st.markdown("---")

    if user_role == "👤 Người Dùng (User)":
        st.subheader("🎬 CHỨC NĂNG NGƯỜI DÙNG")
        app_mode = st.radio(
            "Lựa chọn phân hệ:",
            [
                "🤖 Tìm Phim Qua Mô Tả Cốt Truyện (Semantic AI)",
                "🧪 Pha Chế Điện Ảnh (Cinematic Alchemist)",
                "👤 Cá nhân hóa AI (NCF)",
                "🔍 Tìm Phim Qua Từ Khóa/Tên Phim",
                "🏷️ Tìm Phim Theo Thể Loại"
            ]
        )
        st.markdown("---")
        top_k = st.slider("🍿 Số lượng phim hiển thị (Top K):", min_value=5, max_value=40, value=10)

        st.markdown("---")
        st.subheader("👤 QUẢN LÝ TÀI KHOẢN")
        selected_user_id = st.number_input(
            "🆔 Đổi User ID Hiện Tại:",
            min_value=0, max_value=85306,
            value=st.session_state.current_user_id, step=1
        )
        st.session_state.current_user_id = selected_user_id
        cur_u_data = get_current_user_data(selected_user_id)

        with st.expander(f"📌 Watchlist ({len(cur_u_data['watchlist'])})"):
            if not cur_u_data["watchlist"]:
                st.caption("Chưa lưu phim nào.")
            else:
                for saved_m in list(cur_u_data["watchlist"]):
                    c1, c2 = st.columns([4, 1])
                    c1.write(f"• {saved_m}")
                    clean_saved_key = re.sub(r'[^a-zA-Z0-9_]', '_', saved_m)
                    if c2.button("❌", key=f"del_{clean_saved_key}"):
                        cur_u_data["watchlist"].remove(saved_m)
                        st.rerun()

        with st.expander(f"📜 Lịch sử xem ({len(cur_u_data['history'])})"):
            for hist_m in reversed(cur_u_data["history"]):
                st.caption(f"👀 {hist_m}")

    else:  # Admin Role
        st.subheader("⚙️ BẢNG ĐIỀU HÀNH ADMIN")
        app_mode = st.radio(
            "Lựa chọn phân hệ quản trị:",
            [
                "🌌 Đồ thị Tri thức (Knowledge Graph)",
                "🔬 ML Lab & Active Learning",
                "📊 Giám Sát Hệ Thống & Dữ Liệu"
            ]
        )

# ---------------------------------------------------------
# HEADER HỆ THỐNG
# ---------------------------------------------------------
st.title("🎬 NETFLIX ENTERPRISE SYSTEMS - V6.5 AI MASTER")
st.caption(f"Đang hoạt động dưới quyền: **{user_role}**")

# ---------------------------------------------------------
# PHÂN HỆ NGƯỜI DÙNG (USER MODES)
# ---------------------------------------------------------
if user_role == "👤 Người Dùng (User)":

    if app_mode == "🤖 Tìm Phim Qua Mô Tả Cốt Truyện (Semantic AI)":
        st.header("🤖 Multilingual Plot Semantic Search (Tìm Phim Qua Mô Tả)")
        st.markdown("💡 **SBERT Đa Ngôn Ngữ:** Phân tích trực tiếp Tiếng Việt & Tiếng Anh bằng AI ngữ nghĩa chuyên sâu.")

        plot_query = st.text_area(
            "📝 Nhập câu mô tả nội dung, bối cảnh, nhân vật hoặc diễn biến phim bạn muốn tìm:",
            placeholder="Ví dụ: Một cậu bé phù thủy vô tình tham gia cuộc thi nguy hiểm với rồng...", height=110
        )

        if st.button("🔍 GIẢI MÃ CỐT TRUYỆN & TÌM PHIM"):
            if not plot_query.strip():
                st.error("Vui lòng nhập nội dung mô tả bộ phim!")
            else:
                with st.spinner("🧠 AI đang phân tích ngữ nghĩa cốt truyện..."):
                    start_perf = time.perf_counter()
                    search_query_en = plot_query
                    is_vietnamese = bool(
                        re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
                                  plot_query.lower()))

                    if HAS_SBERT and overview_embeddings is not None:
                        query_vec = sbert_model.encode([plot_query], convert_to_numpy=True)
                        dense_scores = cosine_similarity(query_vec, overview_embeddings).flatten()
                    else:
                        dense_scores = np.zeros(len(movies_df))
                        if is_vietnamese:
                            translated_txt, success = safe_translate_text(plot_query)
                            if success: search_query_en = translated_txt

                    tfidf_query = tfidf_vectorizer.transform([search_query_en])
                    sparse_scores = cosine_similarity(tfidf_query, tfidf_matrix).flatten()

                    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', search_query_en.lower())
                    stopwords = {'this', 'film', 'series', 'set', 'and', 'follows', 'the', 'for', 'with', 'movie',
                                 'story'}
                    keywords = [w for w in words if w not in stopwords]

                    boost_scores = np.zeros(len(movies_df))
                    if keywords:
                        for kw in keywords:
                            title_match = movies_df['title_lower'].str.contains(kw, regex=False).values
                            boost_scores += (title_match.astype(float) * 0.1)
                        max_b = np.max(boost_scores)
                        if max_b > 0: boost_scores = boost_scores / max_b

                    if HAS_SBERT and overview_embeddings is not None:
                        final_scores = (0.75 * dense_scores) + (0.15 * sparse_scores) + (0.10 * boost_scores)
                    else:
                        final_scores = (0.70 * sparse_scores) + (0.30 * boost_scores)

                    exclude_norm_titles = get_excluded_norm_titles(st.session_state.current_user_id)
                    top_indices = np.argsort(final_scores)[::-1]

                    raw_recs = []
                    seen_in_session = set(exclude_norm_titles)

                    for idx in top_indices:
                        m_info = movies_df.iloc[idx].to_dict()
                        norm_t = m_info['norm_title']
                        if norm_t in seen_in_session: continue

                        seen_in_session.add(norm_t)
                        match_score = min(99.9, max(50.0, final_scores[idx] * 100))
                        extra_html = f'<div style="margin-bottom:6px;"><span style="font-size:12px; color:#2ed573; font-weight:bold;">🎯 ĐỘ KHỚP HYBRID: {match_score:.1f}%</span></div>'

                        m_info['genres_raw'] = m_info['genres_list']
                        m_info['extra_html'] = extra_html
                        raw_recs.append(m_info)
                        if len(raw_recs) == top_k: break

                    recs = resolve_movies_batch(raw_recs)
                    end_perf = time.perf_counter()
                    st.session_state.recommendations = {'mode': 'plot', 'recs': recs,
                                                        'time': (end_perf - start_perf) * 1000}

        if st.session_state.recommendations and st.session_state.recommendations.get('mode') == 'plot':
            rec_data = st.session_state.recommendations
            st.success(f"🎉 Phân tích hoàn tất trong **{rec_data['time']:.2f} ms**!")
            cols = st.columns(5)
            for i, item in enumerate(rec_data['recs'][:top_k]):
                render_movie_card(cols[i % 5], item['title'], item['genres_raw'], item['overview_text'],
                                  item['poster_url'], item['extra_html'], key_prefix="plot")

    elif app_mode == "🧪 Pha Chế Điện Ảnh (Cinematic Alchemist)":
        st.header("🧪 Cinematic Alchemist Engine (Thuật Toán Pha Chế Vector)")

        col_blend1, col_blend2 = st.columns(2)
        with col_blend1:
            movie_a_title = st.selectbox("🎬 Phim A:", movies_df['title'].values, index=0)
        with col_blend2:
            movie_b_title = st.selectbox("🎭 Phim B:", movies_df['title'].values, index=min(1, len(movies_df) - 1))

        ratio_a = st.slider("⚖️ Tỷ lệ pha trộn (Phim A đóng góp):", 10, 90, 50, step=5, format="%d%%")

        if st.button("🧪 THỰC HIỆN HÒA TRỘN VECTOR"):
            if movie_a_title == movie_b_title:
                st.error("Vui lòng chọn 2 bộ phim khác nhau!")
            else:
                with st.spinner("🔬 Đang tính toán Vector Embeddings..."):
                    start_perf = time.perf_counter()
                    idx_a = movies_df[movies_df['title'] == movie_a_title].index[0]
                    idx_b = movies_df[movies_df['title'] == movie_b_title].index[0]

                    model.eval()
                    with torch.no_grad():
                        movie_embeddings = model.movie_embedding.weight.detach().cpu().numpy()

                    vec_a = movie_embeddings[idx_a % model.num_movies]
                    vec_b = movie_embeddings[idx_b % model.num_movies]

                    w_a = ratio_a / 100.0
                    vec_blended = (w_a * vec_a) + ((1.0 - w_a) * vec_b)
                    blended_sims = cosine_similarity(vec_blended.reshape(1, -1), movie_embeddings).flatten()

                    exclude_norm_titles = get_excluded_norm_titles(st.session_state.current_user_id)
                    exclude_norm_titles.add(normalize_title(movie_a_title))
                    exclude_norm_titles.add(normalize_title(movie_b_title))

                    sorted_idxs = np.argsort(blended_sims)[::-1]
                    seen_in_session = set(exclude_norm_titles)

                    raw_recs = []
                    for i in sorted_idxs:
                        m_info = movies_df.iloc[i].to_dict()
                        norm_t = m_info['norm_title']
                        if norm_t not in seen_in_session:
                            seen_in_session.add(norm_t)
                            match_percent = blended_sims[i] * 100
                            extra_html = f'<div style="margin-bottom:6px;"><span style="font-size:12px; color:#E50914; font-weight:bold;">🧪 ĐỘ HÒA HỢP: {match_percent:.1f}%</span></div>'
                            m_info['genres_raw'] = m_info['genres_list']
                            m_info['extra_html'] = extra_html
                            raw_recs.append(m_info)
                        if len(raw_recs) == top_k: break

                    recs = resolve_movies_batch(raw_recs)
                    end_perf = time.perf_counter()
                    st.session_state.recommendations = {'mode': 'blend', 'recs': recs,
                                                        'time': (end_perf - start_perf) * 1000}

        if st.session_state.recommendations and st.session_state.recommendations.get('mode') == 'blend':
            rec_data = st.session_state.recommendations
            cols = st.columns(5)
            for i, item in enumerate(rec_data['recs'][:top_k]):
                render_movie_card(cols[i % 5], item['title'], item['genres_raw'], item['overview_text'],
                                  item['poster_url'], item['extra_html'], key_prefix="blend")

    elif app_mode == "👤 Cá nhân hóa AI (NCF)":
        st.header(f"🎯 Gợi ý cá nhân hóa Deep Learning (NCF) cho User ID: {st.session_state.current_user_id}")
        selected_genre = st.selectbox("Lọc nhanh theo thể loại:", ["Tất cả thể loại"] + all_genres)

        cur_uid = st.session_state.current_user_id
        u_data = get_current_user_data(cur_uid)
        filter_watched = st.checkbox("🛡️ Bỏ qua phim đã nằm trong danh sách Xem/Lưu", value=True)

        if st.button("🚀 XUẤT GỢI Ý CÁ NHÂN HÓA"):
            with st.spinner('Đang tính toán Dynamic User Vector & chạy Neural Network...'):
                start_perf = time.perf_counter()
                model.eval()

                user_interacted_movies = list(set(u_data["watchlist"] + u_data["history"]))
                dynamic_user_emb = get_dynamic_user_embedding(model, cur_uid, movies_df, user_interacted_movies,
                                                              alpha=0.4)
                movie_tensor = torch.arange(len(movies_df), dtype=torch.long).to(device)

                with torch.no_grad():
                    scores = predict_with_custom_user_embed(model, dynamic_user_emb, movie_tensor).cpu().numpy()

                all_indices = np.argsort(scores)[::-1]
                exclude_norm_titles = get_excluded_norm_titles(cur_uid) if filter_watched else set()
                seen_in_session = set(exclude_norm_titles)

                raw_recs = []
                for idx in all_indices:
                    m_info = movies_df.iloc[idx].to_dict()
                    norm_t = m_info['norm_title']

                    if filter_watched and norm_t in seen_in_session: continue

                    if selected_genre == "Tất cả thể loại" or selected_genre in m_info['genres_list']:
                        seen_in_session.add(norm_t)
                        extra_html = f'<div style="margin-bottom:4px;"><span style="font-size:12px; color:#E50914; font-weight:bold;">🔥 MATCH NCF: {scores[idx] * 100:.1f}%</span></div>'
                        m_info['genres_raw'] = m_info['genres_list']
                        m_info['extra_html'] = extra_html
                        raw_recs.append(m_info)

                    if len(raw_recs) == top_k: break

                recs = resolve_movies_batch(raw_recs)
                end_perf = time.perf_counter()
                st.session_state.recommendations = {'mode': 'ncf', 'recs': recs, 'time': (end_perf - start_perf) * 1000}

        if st.session_state.recommendations and st.session_state.recommendations.get('mode') == 'ncf':
            rec_data = st.session_state.recommendations
            cols = st.columns(5)
            for i, item in enumerate(rec_data['recs'][:top_k]):
                render_movie_card(cols[i % 5], item['title'], item['genres_raw'], item['overview_text'],
                                  item['poster_url'], item['extra_html'], key_prefix="ncf")

    elif app_mode == "🔍 Tìm Phim Qua Từ Khóa/Tên Phim":
        st.header("🔍 Tìm Phim Trực Tiếp Theo Tên Phim / Từ Khóa")
        search_query = st.text_input("🔎 Nhập tên phim hoặc từ khóa:")

        if st.button("🔍 TÌM KIẾM PHIM"):
            if search_query.strip():
                query_clean = search_query.strip().lower()
                matched_df = movies_df[movies_df['title_lower'].str.contains(query_clean, regex=False)]

                raw_recs = []
                seen_in_session = set()
                for _, m_row in matched_df.iterrows():
                    m_info = m_row.to_dict()
                    norm_t = m_info['norm_title']
                    if norm_t in seen_in_session: continue
                    seen_in_session.add(norm_t)

                    m_info['genres_raw'] = m_info['genres_list']
                    m_info['extra_html'] = ''
                    raw_recs.append(m_info)
                    if len(raw_recs) == top_k: break
                recs = resolve_movies_batch(raw_recs)
                st.session_state.recommendations = {'mode': 'search', 'recs': recs}

        if st.session_state.recommendations and st.session_state.recommendations.get('mode') == 'search':
            cols = st.columns(5)
            for i, item in enumerate(st.session_state.recommendations['recs'][:top_k]):
                render_movie_card(cols[i % 5], item['title'], item['genres_raw'], item['overview_text'],
                                  item['poster_url'], item['extra_html'], key_prefix="search")

    elif app_mode == "🏷️ Tìm Phim Theo Thể Loại":
        st.header("🏷️ Khám Phá Điện Ảnh Theo Thể Loại")

        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            selected_genres = st.multiselect("Chọn thể loại phim:", all_genres,
                                             default=[all_genres[0]] if all_genres else [])
        with col_g2:
            filter_logic = st.radio("Logic tìm kiếm:", ["HOẶC (OR)", "VÀ (AND)"])

        exclude_norm_titles = get_excluded_norm_titles(st.session_state.current_user_id)
        seen_in_session = set(exclude_norm_titles)

        raw_recs = []
        if selected_genres:
            for idx, row in movies_df.iterrows():
                norm_t = row['norm_title']
                if norm_t in seen_in_session: continue

                row_genres = row['genres_list']
                match = any(g in row_genres for g in selected_genres) if filter_logic == "HOẶC (OR)" else all(
                    g in row_genres for g in selected_genres)

                if match:
                    seen_in_session.add(norm_t)
                    m_info = row.to_dict()
                    m_info['genres_raw'] = m_info['genres_list']
                    m_info['extra_html'] = ''
                    raw_recs.append(m_info)

                if len(raw_recs) == top_k: break

        recs = resolve_movies_batch(raw_recs)
        cols = st.columns(5)
        for i, item in enumerate(recs):
            render_movie_card(cols[i % 5], item['title'], item['genres_raw'], item['overview_text'], item['poster_url'],
                              key_prefix="genre")

# ---------------------------------------------------------
# PHÂN HỆ QUẢN TRỊ VIÊN (ADMIN MODES)
# ---------------------------------------------------------
else:
    if app_mode == "🌌 Đồ thị Tri thức (Knowledge Graph)":
        st.header("🌌 Neural Knowledge Graph - Phân Tích Vũ Trụ Điện Ảnh")
        st.caption("Công cụ quản trị viên dùng để trực quan hóa cấu trúc liên kết embedding giữa các bộ phim.")

        if not HAS_PYVIS:
            st.error("⚠️ Cần cài đặt pyvis: `pip install pyvis`!")
        else:
            root_movie = st.selectbox("🎯 Chọn bộ phim làm Tâm điểm (Root Node):", movies_df['title'].values)
            num_neighbors = st.slider("Mật độ liên kết (Neighbors):", 5, 30, 15)

            if st.button("🚀 KHỞI TẠO KHÔNG GIAN ĐỒ THỊ"):
                query_idx = movies_df[movies_df['title'] == root_movie].index[0]
                model.eval()
                with torch.no_grad():
                    movie_embeddings = model.movie_embedding.weight.detach().cpu().numpy()

                query_dense = movie_embeddings[query_idx % model.num_movies].reshape(1, -1)
                sims = cosine_similarity(query_dense, movie_embeddings).flatten()
                top_neighbor_idxs = np.argsort(sims)[::-1][1:num_neighbors + 1]

                net = Network(height="600px", width="100%", bgcolor="#050508", font_color="white")
                net.add_node(root_movie, label=root_movie, color="#E50914", size=25)

                seen_nodes = {normalize_title(root_movie)}
                for n_idx in top_neighbor_idxs:
                    m_row = movies_df.iloc[n_idx]
                    m_title = m_row['title']
                    norm_t = m_row['norm_title']

                    if norm_t not in seen_nodes:
                        seen_nodes.add(norm_t)
                        sim_val = sims[n_idx]
                        net.add_node(m_title, label=m_title, color="#00d2d3", size=15)
                        net.add_edge(root_movie, m_title, value=float(sim_val))

                html_content = net.generate_html()
                components.html(html_content, height=620, scrolling=False)

    elif app_mode == "🔬 ML Lab & Active Learning":
        st.header("🔬 ML Lab & Active Learning Studio (Dành cho Admin)")
        st.caption("Huấn luyện On-the-fly mạng NCF dựa trên phản hồi đánh giá thực tế của người dùng.")

        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            optimizer_name = st.selectbox("⚙️ Thuật toán tối ưu (Optimizer):", ["Adam", "AdamW", "SGD"])
            learning_rate = st.select_slider("⚡ Learning Rate:", options=[0.0001, 0.001, 0.005, 0.01], value=0.001)
        with col_lab2:
            epochs = st.slider("🔄 Số lượng Epochs:", 1, 10, 3)
            st.markdown(
                f"📊 **Dữ liệu đánh giá người dùng (Active Feedback):** {len(st.session_state.user_ratings)} bộ phim")

        if st.button("🚀 KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN REAL-TIME"):
            if not st.session_state.user_ratings:
                st.warning("Vui lòng đánh giá (⭐ 1-5 Sao) ít nhất 1 bộ phim trong các tab gợi ý trước khi huấn luyện!")
            else:
                model.train()
                if optimizer_name == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                elif optimizer_name == "AdamW":
                    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                criterion = nn.BCELoss()
                loss_history = []

                rated_titles = list(st.session_state.user_ratings.keys())
                rated_indices = movies_df[movies_df['title'].isin(rated_titles)].index.tolist()
                user_idx_tensor = torch.tensor([st.session_state.current_user_id] * len(rated_indices),
                                               dtype=torch.long).to(device)
                movie_idx_tensor = torch.tensor(rated_indices, dtype=torch.long).to(device)
                labels = torch.tensor(
                    [st.session_state.user_ratings[movies_df.iloc[idx]['title']] / 5.0 for idx in rated_indices],
                    dtype=torch.float32).to(device)

                progress_bar = st.progress(0)
                status_text = st.empty()

                for ep in range(epochs):
                    optimizer.zero_grad()
                    preds = model(user_idx_tensor, movie_idx_tensor).view(-1)
                    labels_tensor = labels.view(-1)
                    loss = criterion(preds, labels_tensor)
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    loss_history.append(loss_val)
                    progress_bar.progress((ep + 1) / epochs)
                    status_text.text(f"Epoch {ep + 1}/{epochs} - Loss: {loss_val:.4f}")
                    time.sleep(0.1)

                st.success("✅ Huấn luyện hoàn tất! Trực quan hóa giá trị Loss (Gradient Descent):")
                st.line_chart(pd.DataFrame({"Loss": loss_history}))

    elif app_mode == "📊 Giám Sát Hệ Thống & Dữ Liệu":
        st.header("📊 Admin Dashboard & Telemetry")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="🎥 Tổng số Phim trong DB", value=f"{len(movies_df):,}")
        with m_col2:
            st.metric(label="⭐ Ratings đã ghi nhận", value=len(st.session_state.user_ratings))
        with m_col3:
            st.metric(label="🖥️ Phần cứng AI", value=str(device).upper())
        with m_col4:
            st.metric(label="👥 Active Sessions", value=len(st.session_state.user_store))

        st.markdown("---")
        st.subheader("📋 Dữ Liệu Ratings Gợi Ý Thực Tế (Active Feedback)")
        if st.session_state.user_ratings:
            ratings_df = pd.DataFrame(list(st.session_state.user_ratings.items()), columns=["Tên Phim", "Số Sao (1-5)"])
            st.dataframe(ratings_df, use_container_width=True)
        else:
            st.info("Chưa có lượt đánh giá phim nào từ phía người dùng.")