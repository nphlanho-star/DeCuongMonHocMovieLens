import os
import sys
import warnings
import logging
import io
import json
import time
import re
import html
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

    /* Ép kiểu chữ mô tả đồng nhất, không bị lỗi font to */
    .overview-text, .overview-text p { 
        font-size: 11px !important; 
        font-weight: normal !important;
        color: #aaaaaa !important; 
        height: 42px !important; 
        overflow: hidden !important; 
        display: -webkit-box !important; 
        -webkit-line-clamp: 3 !important; 
        -webkit-box-orient: vertical !important; 
        margin-top: 5px !important; 
        line-height: 1.3 !important;
    }

    .inspector-card {
        background-color: #12121a;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 13px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .inspector-card:hover { transform: translateX(4px); }
    .inspector-watchlist { border-left: 4px solid #3b82f6; }
    .inspector-history { border-left: 4px solid #10b981; }
    .inspector-rating { border-left: 4px solid #f59e0b; }
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


def clean_overview_for_display(text):
    if not text or not isinstance(text, str):
        return "No overview available."
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^[■📖\s]+', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


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
    if not isinstance(movie_title, str): return ""
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
        "http") and poster_url_from_df != FALLBACK_IMAGE
    ) else FALLBACK_IMAGE
    fetched_overview = None

    try:
        response = requests.get(url, headers=headers, timeout=2.5)
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
        res = requests.get(search_url, headers=headers, timeout=1.5)
        if res.status_code == 200:
            results = res.json().get('results', [])
            if results:
                movie_id = results[0]['id']
                videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
                v_res = requests.get(videos_url, headers=headers, timeout=1.5)
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
    if os.path.exists(csv_path):
        movies_df = pd.read_csv(csv_path)
    else:
        data = {
            'movieId': [1, 2, 3],
            'title': ['The Dark Knight', 'Inception', 'Interstellar'],
            'genres': ['Action|Crime', 'Action|Sci-Fi', 'Adventure|Drama'],
            'overview': ['Batman fights Joker in Gotham City.', 'A thief enters dreams to steal secrets.',
                         'Explorers travel through a wormhole in space.']
        }
        movies_df = pd.DataFrame(data)

    if 'overview' not in movies_df.columns: movies_df['overview'] = "No overview available."
    if 'poster_url' not in movies_df.columns: movies_df['poster_url'] = FALLBACK_IMAGE

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
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# ---------------------------------------------------------
# DYNAMIC SESSION STATE & CÁC HÀM CẬP NHẬT LŨY THỪA SIÊU TỐC
# ---------------------------------------------------------
if "working_movies_df" not in st.session_state:
    st.session_state.working_movies_df = movies_df.copy()
if "working_tfidf_matrix" not in st.session_state:
    st.session_state.working_tfidf_matrix = tfidf_matrix
if "working_overview_embeddings" not in st.session_state:
    st.session_state.working_overview_embeddings = overview_embeddings


def refit_tfidf_fast():
    df = st.session_state.working_movies_df
    enhanced_corpus = (
            df['genres'].fillna('').str.replace('|', ' ', regex=False) + ' ' +
            df['title'].fillna('') + ' ' +
            df['overview'].fillna('')
    )
    st.session_state.working_tfidf_matrix = tfidf_vectorizer.fit_transform(enhanced_corpus)


def save_to_disk():
    csv_path = 'movies_enriched.csv' if os.path.exists('movies_enriched.csv') else 'movies_mapped.csv'
    save_cols = [c for c in ['movieId', 'title', 'genres', 'overview', 'poster_url'] if
                 c in st.session_state.working_movies_df.columns]
    st.session_state.working_movies_df[save_cols].to_csv(csv_path, index=False)

    if HAS_SBERT and st.session_state.working_overview_embeddings is not None:
        np.save('overview_embeddings.npy', st.session_state.working_overview_embeddings)

    st.cache_resource.clear()
    st.cache_data.clear()


def add_movie_fast(new_row_dict):
    new_df = pd.DataFrame([new_row_dict])
    st.session_state.working_movies_df = pd.concat([st.session_state.working_movies_df, new_df], ignore_index=True)

    if HAS_SBERT and sbert_model is not None and st.session_state.working_overview_embeddings is not None:
        new_text = f"{new_row_dict['title']}. {new_row_dict['overview']}"
        new_vec = sbert_model.encode([new_text], convert_to_numpy=True, show_progress_bar=False)
        st.session_state.working_overview_embeddings = np.vstack(
            [st.session_state.working_overview_embeddings, new_vec])

    refit_tfidf_fast()
    save_to_disk()


def edit_movie_fast(idx, title, genres, overview, poster_url):
    df = st.session_state.working_movies_df
    df.at[idx, 'title'] = title
    df.at[idx, 'norm_title'] = normalize_title(title)
    df.at[idx, 'title_lower'] = title.lower()
    df.at[idx, 'genres'] = genres
    df.at[idx, 'genres_list'] = [g for g in genres.split('|') if g]
    df.at[idx, 'overview'] = overview
    df.at[idx, 'overview_str'] = overview
    df.at[idx, 'poster_url'] = poster_url

    if HAS_SBERT and sbert_model is not None and st.session_state.working_overview_embeddings is not None:
        new_text = f"{title}. {overview}"
        new_vec = sbert_model.encode([new_text], convert_to_numpy=True, show_progress_bar=False)
        if idx < len(st.session_state.working_overview_embeddings):
            st.session_state.working_overview_embeddings[idx] = new_vec[0]

    refit_tfidf_fast()
    save_to_disk()


def delete_movie_fast(idx):
    st.session_state.working_movies_df = st.session_state.working_movies_df.drop(idx).reset_index(drop=True)

    if HAS_SBERT and sbert_model is not None and st.session_state.working_overview_embeddings is not None:
        if idx < len(st.session_state.working_overview_embeddings):
            st.session_state.working_overview_embeddings = np.delete(st.session_state.working_overview_embeddings, idx,
                                                                     axis=0)

    refit_tfidf_fast()
    save_to_disk()


active_movies_df = st.session_state.working_movies_df
active_tfidf_matrix = st.session_state.working_tfidf_matrix
active_overview_embeddings = st.session_state.working_overview_embeddings

all_genres = sorted(list(set([g for g_list in active_movies_df['genres_list'] for g in g_list if g])))

# ---------------------------------------------------------
# KHO LƯU TRỮ SESSION STATE (CHUẨN HÓA LOGIC USER & ADMIN)
# ---------------------------------------------------------
if "user_store" not in st.session_state: st.session_state.user_store = {}
if "current_user_id" not in st.session_state: st.session_state.current_user_id = 80000
if "recommendations" not in st.session_state: st.session_state.recommendations = None
if "active_trailer" not in st.session_state: st.session_state.active_trailer = None
if "alpha_blend" not in st.session_state: st.session_state.alpha_blend = 0.4
if "admin_authenticated" not in st.session_state: st.session_state.admin_authenticated = False


def get_current_user_data(uid):
    if uid not in st.session_state.user_store:
        st.session_state.user_store[uid] = {
            "watchlist": [],
            "history": [],
            "ratings": {}
        }
    return st.session_state.user_store[uid]


def get_all_recorded_ratings():
    records = []
    for uid, udata in st.session_state.user_store.items():
        for title, rating in udata.get("ratings", {}).items():
            if rating > 0:
                records.append({"user_id": uid, "title": title, "rating": rating})
    return records


def get_excluded_norm_titles(user_id):
    u_data = get_current_user_data(user_id)
    raw_list = u_data["watchlist"] + u_data["history"]
    return set([normalize_title(t) for t in raw_list if t])


def get_dynamic_user_embedding(model, user_id, df, user_history_and_watchlist, alpha=0.4):
    model.eval()
    with torch.no_grad():
        u_t = torch.tensor([user_id % 85307], dtype=torch.long).to(device)
        base_user_emb = model.user_embedding(u_t).squeeze(0)

        if not user_history_and_watchlist:
            return base_user_emb

        norm_interacted = set([normalize_title(t) for t in user_history_and_watchlist])
        interacted_indices = df[df['norm_title'].isin(norm_interacted)].index.tolist()

        if not interacted_indices:
            return base_user_emb

        m_indices_t = torch.tensor(interacted_indices, dtype=torch.long).to(device)
        m_embeds = model.movie_embedding(m_indices_t)
        history_emb = torch.mean(m_embeds, dim=0)

        dynamic_emb = (alpha * base_user_emb) + ((1.0 - alpha) * history_emb)
        return dynamic_emb


# ---------------------------------------------------------
# KHUNG RENDER CARD PHIM CHUẨN CINEMATIC (ĐÃ SỬA LỖI HTML)
# ---------------------------------------------------------
def render_movie_card(col, title, genres_raw, overview_text, poster_url, extra_info_html="", key_prefix="card"):
    cur_uid = st.session_state.current_user_id
    u_data = get_current_user_data(cur_uid)

    clean_ov = clean_overview_for_display(overview_text)
    attr_ov = html.escape(clean_ov, quote=True)
    display_ov = html.escape(clean_ov)

    with col:
        st.image(poster_url, width="stretch")
        badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw if g])

        st.markdown(
            f'<h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>',
            unsafe_allow_html=True)

        if extra_info_html:
            st.markdown(extra_info_html, unsafe_allow_html=True)

        st.markdown(f'<div>{badges}</div><div class="overview-text" title="{attr_ov}">📖 {display_ov}</div>',
                    unsafe_allow_html=True)

        clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', title)
        card_id = f"{key_prefix}_{clean_key}"

        rating_key = f"rate_{card_id}_{cur_uid}"
        cur_rating = u_data["ratings"].get(title, 0)
        rating = st.selectbox("⭐ Active Learning", [0, 1, 2, 3, 4, 5], index=cur_rating, key=rating_key,
                              help="Đánh giá để huấn luyện mô hình cho User này")
        if rating != cur_rating:
            u_data["ratings"][title] = rating
            if rating >= 4 and title not in u_data["history"]:
                u_data["history"].append(title)
            st.rerun()

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("▶️ Trailer", key=f"tr_{card_id}_{cur_uid}"):
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
            if st.button(heart_icon, key=f"wl_{card_id}_{cur_uid}"):
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
                    if c2.button("❌", key=f"del_{clean_saved_key}_{selected_user_id}"):
                        cur_u_data["watchlist"].remove(saved_m)
                        st.rerun()

        with st.expander(f"📜 Lịch sử xem ({len(cur_u_data['history'])})"):
            for hist_m in reversed(cur_u_data["history"]):
                st.caption(f"👀 {hist_m}")

        with st.expander(f"⭐ Đánh giá của bạn ({len(cur_u_data['ratings'])})"):
            for r_title, r_val in cur_u_data["ratings"].items():
                st.caption(f"🎬 {r_title}: **{r_val}⭐**")

    else:  # Admin Role
        st.subheader("⚙️ XÁC THỰC QUẢN TRỊ VIÊN")
        admin_pass = st.text_input("🔑 Mật khẩu Admin:", type="password")
        if admin_pass == "admin123" or st.session_state.admin_authenticated:
            st.session_state.admin_authenticated = True
            st.success("🔓 Đã đăng nhập Quản Trị Viên")
            app_mode = st.radio(
                "Lựa chọn phân hệ quản trị:",
                [
                    "🎬 Quản Lý Kho Phim (Full CRUD Vector)",
                    "🌌 Đồ thị Tri thức (Knowledge Graph)",
                    "🔬 ML Lab & Active Learning",
                    "📊 Giám Sát Hệ Thống & User Inspector",
                    "📈 Analytics & Xuất Dữ Liệu (Data Export)"
                ]
            )
            st.markdown("---")
            st.subheader("⚙️ CẤU HÌNH THUẬT TOÁN HỆ THỐNG")
            st.session_state.alpha_blend = st.slider(
                "⚖️ Hệ số Alpha NCF (Base vs History):",
                min_value=0.0, max_value=1.0, value=st.session_state.alpha_blend, step=0.05,
                help="1.0 = Hoàn toàn dựa vào ID gốc; 0.0 = Hoàn toàn dựa vào Lịch sử tương tác"
            )
        else:
            st.warning("🔒 Vui lòng nhập mật khẩu `admin123` để truy cập!")
            app_mode = "LOCKED"

# ---------------------------------------------------------
# HEADER HỆ THỐNG
# ---------------------------------------------------------
st.title("🎬 NETFLIX ENTERPRISE SYSTEMS - V6.5 AI MASTER")
st.caption(f"Đang hoạt động dưới quyền: **{user_role}**" + (
    f" | Active User ID: **{st.session_state.current_user_id}**" if user_role == "👤 Người Dùng (User)" else ""))

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

                    if is_vietnamese:
                        translated_txt, success = safe_translate_text(plot_query)
                        if success:
                            search_query_en = translated_txt

                    if HAS_SBERT and active_overview_embeddings is not None:
                        query_vec = sbert_model.encode([search_query_en], convert_to_numpy=True)
                        dense_scores = cosine_similarity(query_vec, active_overview_embeddings).flatten()
                    else:
                        dense_scores = np.zeros(len(active_movies_df))

                    tfidf_query = tfidf_vectorizer.transform([search_query_en])
                    sparse_scores = cosine_similarity(tfidf_query, active_tfidf_matrix).flatten()

                    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', search_query_en.lower())
                    stopwords = {'this', 'film', 'series', 'set', 'and', 'follows', 'the', 'for', 'with', 'movie',
                                 'story', 'about'}
                    keywords = [w for w in words if w not in stopwords]

                    boost_scores = np.zeros(len(active_movies_df))
                    if keywords:
                        for kw in keywords:
                            title_match = active_movies_df['title_lower'].str.contains(kw, regex=False).values
                            boost_scores += (title_match.astype(float) * 0.15)
                        max_b = np.max(boost_scores)
                        if max_b > 0: boost_scores = boost_scores / max_b

                    if HAS_SBERT and active_overview_embeddings is not None:
                        final_scores = (0.70 * dense_scores) + (0.20 * sparse_scores) + (0.10 * boost_scores)
                    else:
                        final_scores = (0.70 * sparse_scores) + (0.30 * boost_scores)

                    exclude_norm_titles = get_excluded_norm_titles(st.session_state.current_user_id)
                    top_indices = np.argsort(final_scores)[::-1]

                    raw_recs = []
                    seen_in_session = set(exclude_norm_titles)

                    for idx in top_indices:
                        if idx >= len(active_movies_df): continue
                        m_info = active_movies_df.iloc[idx].to_dict()
                        norm_t = m_info['norm_title']
                        if norm_t in seen_in_session: continue

                        seen_in_session.add(norm_t)
                        match_score = min(99.9, max(0.0, final_scores[idx] * 100))
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
            movie_a_title = st.selectbox("🎬 Phim A:", active_movies_df['title'].values, index=0)
        with col_blend2:
            movie_b_title = st.selectbox("🎭 Phim B:", active_movies_df['title'].values,
                                         index=min(1, len(active_movies_df) - 1))

        ratio_a = st.slider("⚖️ Tỷ lệ pha trộn (Phim A đóng góp):", 10, 90, 50, step=5, format="%d%%")

        if st.button("🧪 THỰC HIỆN HÒA TRỘN VECTOR"):
            if movie_a_title == movie_b_title:
                st.error("Vui lòng chọn 2 bộ phim khác nhau!")
            else:
                with st.spinner("🔬 Đang tính toán Vector Embeddings..."):
                    start_perf = time.perf_counter()
                    idx_a = active_movies_df[active_movies_df['title'] == movie_a_title].index[0]
                    idx_b = active_movies_df[active_movies_df['title'] == movie_b_title].index[0]

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
                        if i >= len(active_movies_df): continue
                        m_info = active_movies_df.iloc[i].to_dict()
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
        st.header(f"🎯 Gợi ý cá nhân hóa Deep Learning (NCF) - User ID: {st.session_state.current_user_id}")
        selected_genre = st.selectbox("Lọc nhanh theo thể loại:", ["Tất cả thể loại"] + all_genres)

        cur_uid = st.session_state.current_user_id
        u_data = get_current_user_data(cur_uid)
        filter_watched = st.checkbox("🛡️ Bỏ qua phim đã nằm trong danh sách Xem/Lưu", value=True)

        if st.button("🚀 XUẤT GỢI Ý CÁ NHÂN HÓA"):
            with st.spinner('Đang tính toán Dynamic User Vector & chạy Neural Network...'):
                start_perf = time.perf_counter()
                model.eval()

                user_interacted_movies = list(set(u_data["watchlist"] + u_data["history"]))
                dynamic_user_emb = get_dynamic_user_embedding(
                    model, cur_uid, active_movies_df, user_interacted_movies, alpha=st.session_state.alpha_blend
                )
                movie_tensor = torch.arange(len(active_movies_df), dtype=torch.long).to(device)

                with torch.no_grad():
                    scores = predict_with_custom_user_embed(model, dynamic_user_emb, movie_tensor).cpu().numpy()

                all_indices = np.argsort(scores)[::-1]
                exclude_norm_titles = get_excluded_norm_titles(cur_uid) if filter_watched else set()
                seen_in_session = set(exclude_norm_titles)

                raw_recs = []
                for idx in all_indices:
                    if idx >= len(active_movies_df): continue
                    m_info = active_movies_df.iloc[idx].to_dict()
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
                matched_df = active_movies_df[active_movies_df['title_lower'].str.contains(query_clean, regex=False)]

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
            for idx, row in active_movies_df.iterrows():
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
elif user_role == "⚙️ Quản Trị Viên (Admin)" and st.session_state.admin_authenticated:

    if app_mode == "🎬 Quản Lý Kho Phim (Full CRUD Vector)":
        st.header("🎬 CRUD & Vector Updating Engine")
        st.caption(
            "Cập nhật lũy thừa (Incremental Update) - Tự động ghi vĩnh viễn xuống đĩa cứng kể cả khi restart ứng dụng!")

        tab_list, tab_add, tab_edit, tab_delete = st.tabs([
            "🔍 Danh Sách Phim",
            "➕ Thêm Phim",
            "✏️ Sửa Phim",
            "🗑️ Xóa Phim"
        ])

        with tab_list:
            st.subheader(f"Tổng số phim hiện tại: {len(active_movies_df)}")
            st.dataframe(active_movies_df[['movieId', 'title', 'genres', 'overview']], use_container_width=True)

        with tab_add:
            st.subheader("Thêm Phim Mới (Incremental Embeddings)")
            with st.form("add_form_fast", clear_on_submit=True):
                new_title = st.text_input("Tên phim (*)")
                new_genres = st.text_input("Thể loại (ví dụ: Action|Sci-Fi)")
                new_overview = st.text_area("Mô tả (Overview) (*)")
                new_poster = st.text_input("Poster URL (Tùy chọn)")

                submitted = st.form_submit_button("➕ Thêm Phim Siêu Tốc & Lưu Vĩnh Viễn")
                if submitted:
                    if new_title.strip() and new_overview.strip():
                        new_row_dict = {
                            'movieId': int(time.time()),
                            'title': new_title.strip(),
                            'genres': new_genres.strip(),
                            'overview': new_overview.strip(),
                            'poster_url': new_poster.strip() if new_poster.strip() else FALLBACK_IMAGE,
                            'norm_title': normalize_title(new_title.strip()),
                            'title_lower': new_title.strip().lower(),
                            'genres_list': [g for g in new_genres.strip().split('|') if g],
                            'overview_str': new_overview.strip()
                        }
                        add_movie_fast(new_row_dict)
                        st.success(f"⚡ Đã thêm thành công phim '{new_title}' và lưu vào ổ đĩa!")
                        st.rerun()
                    else:
                        st.error("Vui lòng điền đầy đủ Tên phim và Mô tả!")

        with tab_edit:
            st.subheader("Sửa Thông Tin Phim (Ghi đè Vector theo dòng)")
            search_edit = st.text_input("Gõ tên phim cần tìm để sửa:", key="search_edit_fast")

            if search_edit.strip():
                matches = active_movies_df[
                    active_movies_df['title_lower'].str.contains(search_edit.strip().lower(), regex=False)]
                if not matches.empty:
                    selected_movie = st.selectbox("Chọn phim chính xác:", options=matches['title'].tolist(),
                                                  key="select_edit_fast")
                    movie_idx = matches[matches['title'] == selected_movie].index[0]

                    with st.form("edit_form_fast"):
                        edit_title = st.text_input("Tên phim", value=active_movies_df.at[movie_idx, 'title'])
                        edit_genres = st.text_input("Thể loại", value=str(active_movies_df.at[movie_idx, 'genres']))
                        edit_overview = st.text_area("Mô tả", value=str(active_movies_df.at[movie_idx, 'overview']))
                        edit_poster = st.text_input("Poster URL",
                                                    value=str(active_movies_df.at[movie_idx, 'poster_url']))

                        update_btn = st.form_submit_button("💾 Lưu Thay Đổi Siêu Tốc & Lưu Vĩnh Viễn")
                        if update_btn:
                            edit_movie_fast(movie_idx, edit_title.strip(), edit_genres.strip(), edit_overview.strip(),
                                            edit_poster.strip())
                            st.success(f"⚡ Đã cập nhật phim '{edit_title}' và lưu vào ổ đĩa!")
                            st.rerun()
                else:
                    st.warning("Không tìm thấy phim phù hợp.")

        with tab_delete:
            st.subheader("Xóa Phim (Nối bớt Ma trận Vector)")
            search_del = st.text_input("Gõ tên phim cần tìm để xóa:", key="search_del_fast")

            if search_del.strip():
                matches_del = active_movies_df[
                    active_movies_df['title_lower'].str.contains(search_del.strip().lower(), regex=False)]
                if not matches_del.empty:
                    selected_del_title = st.selectbox("Chọn phim cần xóa:", options=matches_del['title'].tolist(),
                                                      key="select_del_fast")
                    del_idx = matches_del[matches_del['title'] == selected_del_title].index[0]

                    if st.button("🔴 Xóa Phim Này Nhanh & Lưu Vĩnh Viễn", type="primary"):
                        delete_movie_fast(del_idx)
                        st.success(f"⚡ Đã xóa thành công phim '{selected_del_title}' và cập nhật ổ đĩa!")
                        st.rerun()
                else:
                    st.warning("Không tìm thấy phim phù hợp.")

    elif app_mode == "🌌 Đồ thị Tri thức (Knowledge Graph)":
        st.header("🌌 Neural Knowledge Graph - Phân Tích Vũ Trụ Điện Ảnh")
        st.caption("Công cụ quản trị viên dùng để trực quan hóa cấu trúc liên kết embedding giữa các bộ phim.")

        if not HAS_PYVIS:
            st.error("⚠️ Cần cài đặt pyvis: `pip install pyvis`!")
        else:
            root_movie = st.selectbox("🎯 Chọn bộ phim làm Tâm điểm (Root Node):", active_movies_df['title'].values)
            num_neighbors = st.slider("Mật độ liên kết (Neighbors):", 5, 30, 15)

            if st.button("🚀 KHỞI TẠO KHÔNG GIAN ĐỒ THỊ"):
                query_idx = active_movies_df[active_movies_df['title'] == root_movie].index[0]
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
                    if n_idx >= len(active_movies_df): continue
                    m_row = active_movies_df.iloc[n_idx]
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
        st.caption("Huấn luyện On-the-fly mạng NCF dựa trên toàn bộ phản hồi đánh giá thực tế của TẤT CẢ Người dùng.")

        all_ratings = get_all_recorded_ratings()

        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            optimizer_name = st.selectbox("⚙️ Thuật toán tối ưu (Optimizer):", ["Adam", "AdamW", "SGD"])
            learning_rate = st.select_slider("⚡ Learning Rate:", options=[0.0001, 0.001, 0.005, 0.01], value=0.001)
        with col_lab2:
            epochs = st.slider("🔄 Số lượng Epochs:", 1, 10, 3)
            st.markdown(
                f"📊 **Dữ liệu đánh giá đa người dùng:** `{len(all_ratings)}` lượt đánh giá từ `{len(st.session_state.user_store)}` User ID.")

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            train_btn = st.button("🚀 KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN MULTI-USER")
        with col_btn2:
            reset_btn = st.button("🔄 Reset Mô Hình Về Mặc Định")

        if reset_btn:
            st.cache_resource.clear()
            st.success("✅ Đã khôi phục trạng thái mô hình ban đầu!")
            st.rerun()

        if train_btn:
            if not all_ratings:
                st.warning("Chưa có lượt đánh giá (⭐ 1-5 Sao) nào từ phía người dùng để huấn luyện!")
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

                u_indices = []
                m_indices = []
                target_labels = []

                for record in all_ratings:
                    u_id = record["user_id"]
                    t_title = record["title"]
                    r_val = record["rating"]

                    m_match = active_movies_df[active_movies_df['title'] == t_title]
                    if not m_match.empty:
                        m_idx = m_match.index[0]
                        u_indices.append(u_id % 85307)
                        m_indices.append(m_idx % model.num_movies)
                        target_labels.append(r_val / 5.0)

                user_idx_tensor = torch.tensor(u_indices, dtype=torch.long).to(device)
                movie_idx_tensor = torch.tensor(m_indices, dtype=torch.long).to(device)
                labels_tensor = torch.tensor(target_labels, dtype=torch.float32).to(device)

                progress_bar = st.progress(0)
                status_text = st.empty()

                for ep in range(epochs):
                    optimizer.zero_grad()
                    preds = model(user_idx_tensor, movie_idx_tensor).view(-1)
                    loss = criterion(preds, labels_tensor)
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    loss_history.append(loss_val)
                    progress_bar.progress((ep + 1) / epochs)
                    status_text.text(f"Epoch {ep + 1}/{epochs} - Loss: {loss_val:.4f}")
                    time.sleep(0.1)

                st.success(f"✅ Huấn luyện thành công trên {len(u_indices)} mẫu dữ liệu!")
                st.line_chart(pd.DataFrame({"Loss": loss_history}))

    elif app_mode == "📊 Giám Sát Hệ Thống & User Inspector":
        st.header("📊 Admin Dashboard & User Inspector")

        all_ratings = get_all_recorded_ratings()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="🎥 Tổng số Phim trong DB", value=f"{len(active_movies_df):,}")
        with m_col2:
            st.metric(label="⭐ Total Active Ratings", value=len(all_ratings))
        with m_col3:
            st.metric(label="🖥️ AI Hardware Acceleration", value=str(device).upper())
        with m_col4:
            st.metric(label="👥 Active User Sessions", value=len(st.session_state.user_store))

        st.markdown("---")
        st.subheader("🔍 TRA CỨU CHI TIẾT NGƯỜI DÙNG (USER INSPECTOR)")

        inspect_uid = st.number_input("🔎 Nhập User ID cần kiểm tra:", min_value=0, max_value=85306,
                                      value=st.session_state.current_user_id, step=1)

        if inspect_uid in st.session_state.user_store:
            target_data = st.session_state.user_store[inspect_uid]
            i_col1, i_col2, i_col3 = st.columns(3)

            with i_col1:
                st.markdown(f"##### 📌 Watchlist ({len(target_data['watchlist'])})")
                if target_data['watchlist']:
                    for item in target_data['watchlist']:
                        st.markdown(f"""
                            <div class="inspector-card inspector-watchlist">
                                📌 <b>{item}</b>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Chưa lưu phim nào trong Watchlist.")

            with i_col2:
                st.markdown(f"##### 📜 Lịch Sử Xem ({len(target_data['history'])})")
                if target_data['history']:
                    for item in reversed(target_data['history']):
                        st.markdown(f"""
                            <div class="inspector-card inspector-history">
                                👀 <b>{item}</b>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Chưa có lịch sử xem phim.")

            with i_col3:
                st.markdown(f"##### ⭐ Đánh Giá ({len(target_data['ratings'])})")
                if target_data['ratings']:
                    for title, r_val in target_data['ratings'].items():
                        stars = "⭐" * r_val
                        st.markdown(f"""
                            <div class="inspector-card inspector-rating">
                                <div style="font-weight: bold; color: #ffffff;">🎬 {title}</div>
                                <div style="color: #f59e0b; margin-top: 4px; font-size: 13px;">{stars} <span style="color: #888; font-size: 11px;">({r_val}/5)</span></div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Chưa đánh giá bộ phim nào.")
        else:
            st.info(f"User ID **{inspect_uid}** chưa phát sinh tương tác trong phiên làm việc này.")

        st.markdown("---")
        st.subheader("📋 BẢNG DỮ LIỆU CÁC LƯỢT ĐÁNH GIÁ (GLOBAL FEEDBACK)")
        if all_ratings:
            ratings_df = pd.DataFrame(all_ratings)
            ratings_df['rating_stars'] = ratings_df['rating'].apply(lambda x: "⭐" * int(x))
            st.dataframe(
                ratings_df[['user_id', 'title', 'rating_stars', 'rating']],
                column_config={
                    "user_id": "User ID",
                    "title": "Tên Phim",
                    "rating_stars": "Đánh Giá (Sao)",
                    "rating": "Điểm Số"
                },
                use_container_width=True
            )
        else:
            st.info("Chưa có lượt đánh giá phim nào được ghi nhận từ phía người dùng.")

    elif app_mode == "📈 Analytics & Xuất Dữ Liệu (Data Export)":
        st.header("📈 Enterprise Analytics & Data Moderation Studio")
        st.caption("Công cụ quản trị chuyên sâu để trích xuất báo cáo dữ liệu và quản lý cache bộ nhớ.")

        tab_analytics, tab_export, tab_maintenance = st.tabs(
            ["📊 Thống Kê & Trend", "📥 Xuất Dữ Liệu CSV/JSON", "🛠️ Quản Lý Cache & System Health"])

        all_ratings = get_all_recorded_ratings()

        with tab_analytics:
            st.subheader("🔥 Top 5 Phim Đánh Giá Cao Nhất Hệ Thống")
            if all_ratings:
                rdf = pd.DataFrame(all_ratings)
                top_m = rdf.groupby('title')['rating'].agg(['count', 'mean']).reset_index()
                top_m = top_m.sort_values(by=['count', 'mean'], ascending=False).head(5)
                top_m.columns = ['Tên Phim', 'Số Lượng Đánh Giá', 'Điểm Trung Bình ⭐']
                st.table(top_m)
            else:
                st.info("Chưa có lượt đánh giá nào để thống kê.")

            st.markdown("---")
            st.subheader("📊 Phân Bố Thể Loại Trong Database")
            genre_counts = {}
            for g_list in active_movies_df['genres_list']:
                for g in g_list:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            gdf = pd.DataFrame(list(genre_counts.items()), columns=['Thể Loại', 'Số Lượng Phim']).sort_values(
                by='Số Lượng Phim', ascending=False)
            st.bar_chart(gdf.set_index('Thể Loại'))

        with tab_export:
            st.subheader("📥 Xuất Báo Cáo Tương Tác Dữ Liệu")
            ex_col1, ex_col2 = st.columns(2)

            with ex_col1:
                st.markdown("##### 📄 Xuất Ratings sang CSV")
                if all_ratings:
                    csv_data = pd.DataFrame(all_ratings).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Tải File ratings_export.csv",
                        data=csv_data,
                        file_name="ratings_export.csv",
                        mime="text/csv"
                    )
                else:
                    st.caption("Chưa có dữ liệu Ratings để tải xuống.")

            with ex_col2:
                st.markdown("##### 📦 Xuất Tất Cả Session Users sang JSON")
                if st.session_state.user_store:
                    json_data = json.dumps(st.session_state.user_store, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Tải File user_sessions.json",
                        data=json_data,
                        file_name="user_sessions.json",
                        mime="application/json"
                    )
                else:
                    st.caption("Chưa có phiên làm việc nào.")

            st.markdown("---")
            st.subheader("🧹 Reset Trạng Thái User ID Cụ Thể")
            reset_uid = st.number_input("Chọn User ID cần xóa dữ liệu:", min_value=0, max_value=85306,
                                        value=st.session_state.current_user_id, step=1)
            if st.button("⚠️ XÓA DỮ LIỆU USER NÀY"):
                if reset_uid in st.session_state.user_store:
                    del st.session_state.user_store[reset_uid]
                    st.success(f"✅ Đã dọn dẹp toàn bộ dữ liệu của User ID {reset_uid}!")
                    st.rerun()
                else:
                    st.info("User ID này không có dữ liệu cần xóa.")

        with tab_maintenance:
            st.subheader("🛠️ Giám Sát Tài Nguyên & Bộ Nhớ Tạm")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("🖥️ VRAM / GPU Device", str(device).upper())
                st.metric("⚡ Memory Embeddings Loaded", f"{len(active_movies_df):,} Vectors")
            with m2:
                if st.button("🧹 Làm Sạch Cache Streamlit (Clear Cache)"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("✅ Đã làm sạch toàn bộ Cache hệ thống!")
                    st.rerun()