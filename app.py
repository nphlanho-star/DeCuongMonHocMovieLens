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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA  # Hỗ trợ giảm chiều không gian vector trực quan

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
# QUẢN LÝ PHẦN CỨNG VẬT LÝ (HARDWARE ACCELERATION ORCHESTRATOR)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# GIAO DIỆN CINEMATIC PREMIUM & HIỆU ỨNG CARD GLASSMORPHISM
# ==============================================================================
st.set_page_config(page_title="Netflix Enterprise v6.0 Academic Master", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050508; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #888888; background-color: transparent; font-weight: bold; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #E50914 !important; border-bottom-color: #E50914 !important; }

    /* Thiết kế Glassmorphism Card đổ bóng 3D */
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
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# ĐỊNH NGHĨA KIẾN TRÚC MẠNG NEURAL NCF SÂU
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


# Hàm hỗ trợ bypass embedding của User để truyền trực tiếp vector tổng hợp (Cold-Start)
def predict_with_custom_user_embed(model, custom_user_embed, movie_indices_tensor):
    model.eval()
    movie_embed = model.movie_embedding(movie_indices_tensor)  # Shape: (N, 32)
    user_embed_expanded = custom_user_embed.unsqueeze(0).expand(movie_embed.size(0), -1)  # Shape: (N, 32)
    x = torch.cat([user_embed_expanded, movie_embed], dim=-1)  # Shape: (N, 64)
    return model.fc_layers(x).squeeze()


def get_movie_poster(movie_title):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    clean_title = movie_title.split(' (')[0].strip()
    if ", The" in clean_title:
        clean_title = "The " + clean_title.replace(", The", "").strip()
    elif ", A" in clean_title:
        clean_title = "A " + clean_title.replace(", A", "").strip()
    elif ", An" in clean_title:
        clean_title = "An " + clean_title.replace(", An", "").strip()

    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={requests.utils.quote(clean_title)}"
    try:
        response = requests.get(url, timeout=1.5).json()
        if response and 'results' in response and len(response['results']) > 0:
            poster_path = response['results'][0]['poster_path']
            if poster_path: return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"


@st.cache_resource
def load_resources():
    model = NeuralCollaborativeFiltering()
    model.load_state_dict(torch.load('ncf_movie_recommender.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    movies_df = pd.read_csv('movies_mapped.csv')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].str.replace('|', ' ') + ' ' + movies_df['title'])

    return model, movies_df, tfidf_matrix


model, movies_df, tfidf_matrix = load_resources()
all_genres = sorted(list(set([g for g_list in movies_df['genres'].str.split('|') for g in g_list])))

# --- DASHBOARD CONTROL PANEL ---
st.title("🎬 NETFLIX ENTERPRISE SYSTEMS - V6.0 MASTER ENGINE")
st.caption(
    "Kiến trúc công nghiệp: Thực thi thiết bị biến ngẫu | Bayesian MC-Dropout | Cold-Start Synth | Explainable AI (XAI)")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1: st.metric(label="🖥️ Thiết bị tính toán tối ưu", value=str(device).upper())
with m_col2: st.metric(label="📊 Chiều sâu Vector Khai triển", value="32-Dimensions")
with m_col3: st.metric(label="🧠 Động cơ Mạng Neural", value="NCF Multi-Layer")
with m_col4: st.metric(label="🔬 Bộ máy Diễn giải XAI", value="Cosine Alignment")

st.markdown("---")

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# --- BỘ ĐIỀU PHỐI TẠI THANH SIDEBAR ---
with st.sidebar:
    st.header("⚙️ CHẾ ĐỘ ENGINE GỢI Ý")
    app_mode = st.radio(
        "Lựa chọn phân hệ chức năng:",
        [
            "👤 Cá nhân hóa AI (NCF)",
            "🔍 Tìm Phim Qua Từ Khóa/Tên (NLP)",
            "🏷️ Tìm Phim Theo Thể Loại",
            "🔬 Phòng Thí Nghiệm Học Máy (Train AI)"
        ]
    )
    st.markdown("---")
    top_k = st.slider("🍿 Số lượng phim hiển thị (Top K):", min_value=5, max_value=20, value=10)

# ==============================================================================
# PHÂN HỆ 1: CÁ NHÂN HÓA QUA MẠNG NEURAL (BAYESIAN MC-DROPOUT & COLD-START)
# ==============================================================================
if app_mode == "👤 Cá nhân hóa AI (NCF)":
    st.header("🎯 Gợi ý cá nhân hóa Deep Learning (Bayesian NCF)")
    st.markdown(
        "💡 **Explainable AI (XAI):** Hệ thống tích hợp khả năng tính toán độ tương đồng Cosine của Vector nhúng để giải thích lý do gợi ý.")

    # 🆕 TÍNH NĂNG COLD-START
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
        cold_start_movies = st.multiselect("Chọn tối thiểu 3 bộ phim bạn yêu thích nhất để tạo lập Hồ sơ số:",
                                           movies_df['title'].values)
        selected_genre = st.selectbox("Lọc nhanh theo dòng phim:", ["Tất cả thể loại"] + all_genres)

    if st.button("🚀 TRUY XUẤT HỒ SƠ VECTOR"):
        if cold_start_mode and len(cold_start_movies) < 3:
            st.error("Vui lòng chọn ít nhất 3 bộ phim để thuật toán Cold-Start phân tích chính xác hành vi.")
        else:
            with st.spinner('Hạ tầng đang phân tích Vector Hành vi...'):
                start_perf = time.perf_counter()

                # CHUẨN BỊ VECTOR NGƯỜI DÙNG (THƯỜNG HOẶC COLD-START)
                model.train()  # Bật Train mode để kích hoạt MC-Dropout
                movie_tensor = torch.tensor(list(range(10524)), dtype=torch.long).to(device)

                if cold_start_mode:
                    # Lấy embedding của các phim người dùng chọn và tính trung bình cộng (Synthesis)
                    selected_indices = [movies_df[movies_df['title'] == title].iloc[0]['movie_idx'] for title in
                                        cold_start_movies]
                    selected_indices_t = torch.tensor(selected_indices, dtype=torch.long).to(device)
                    with torch.no_grad():
                        movie_embs = model.movie_embedding(selected_indices_t)
                        synthetic_user_emb = torch.mean(movie_embs, dim=0)  # Vector 32 chiều
                else:
                    # Lấy embedding của người dùng có sẵn trong DB
                    with torch.no_grad():
                        user_t_single = torch.tensor([user_id_input], dtype=torch.long).to(device)
                        synthetic_user_emb = model.user_embedding(user_t_single).squeeze(0)

                # THỰC THI MC-DROPOUT 5 LẦN
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

                # 🆕 GIẢI THÍCH MÔ HÌNH (XAI)
                # Tính độ tương hợp hướng (Cosine Similarity) giữa vector người dùng và vector tất cả các phim
                with torch.no_grad():
                    all_movie_embs = model.movie_embedding(movie_tensor)
                    u_norm = synthetic_user_emb / synthetic_user_emb.norm(dim=-1, keepdim=True)
                    m_norm = all_movie_embs / all_movie_embs.norm(dim=-1, keepdim=True)
                    alignments = torch.matmul(m_norm, u_norm).cpu().numpy()  # Độ tương đồng Cosine [-1, 1]

                for idx in all_indices:
                    movie_info = movies_df[movies_df['movie_idx'] == idx].iloc[0]
                    if selected_genre == "Tất cả thể loại" or selected_genre in movie_info['genres'].split('|'):
                        valid_indices.append(idx)
                        valid_scores.append(mean_scores[idx])
                        valid_std.append(std_scores[idx])
                        valid_alignments.append(alignments[idx])
                    if len(valid_indices) == top_k: break

                end_perf = time.perf_counter()
                st.markdown(
                    f"⏱️ **Hồ sơ hiệu năng (System Profile):** Tổng hợp hồ sơ mất **{(end_perf - start_perf) * 1000:.2f} ms** trên cấu trúc **{str(device).upper()}**.")

                if not valid_indices:
                    st.warning("Không tìm thấy bộ phim nào phù hợp.")
                else:
                    cols = st.columns(5)
                    for i, idx in enumerate(valid_indices):
                        movie_info = movies_df[movies_df['movie_idx'] == idx].iloc[0]
                        title = movie_info['title']
                        genres_raw = movie_info['genres'].split('|')
                        poster_url = get_movie_poster(title)

                        uncertainty = valid_std[i]
                        if uncertainty < 0.05:
                            confidence_badge = f'<span class="metric-badge badge-success">Tin cậy cao (σ={uncertainty:.3f})</span>'
                        else:
                            confidence_badge = f'<span class="metric-badge badge-warning">Cần khám phá (σ={uncertainty:.3f})</span>'

                        # Chuẩn hóa độ alignment về phần trăm
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
# PHÂN HỆ 2: TÌM KIẾM PHIM THEO TÊN (DENSE-SPARSE HYBRID SEARCH SYSTEM)
# ==============================================================================
elif app_mode == "🔍 Tìm Phim Qua Từ Khóa/Tên (NLP)":
    st.header("🔍 Dense-Sparse Hybrid Semantic Search Engine")
    st.markdown(
        "💡 **Công nghệ Độc quyền:** Phối hợp logic toán học của **TF-IDF (Sparse)** và **Trọng số nhúng mạng Neural (Dense Latent Similarity)**.")

    search_query = st.selectbox("Chọn bộ phim gốc làm tâm điểm dữ liệu:", movies_df['title'].values)
    beta = st.slider("Hệ số lai cấu trúc (Beta - β):", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.caption(
        "💡 *β = 1.0: Sử dụng 100% TF-IDF (Từ ngữ và thể loại) | β = 0.0: Sử dụng 100% Dense Vector (Hành vi tiềm ẩn)*")

    if st.button("🔍 PHÂN TÍCH ĐẶC TRƯNG TIỀM ẨN"):
        start_perf = time.perf_counter()
        query_idx = movies_df[movies_df['title'] == search_query].index[0]

        # 1. Sparse Similarity (Cosine trên ma trận TF-IDF)
        sparse_sim = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()

        # 2. Dense Similarity (Cosine trên Vector Nhúng của mô hình PyTorch)
        model.eval()
        with torch.no_grad():
            movie_embeddings = model.movie_embedding.weight.detach().cpu().numpy()
        query_dense = movie_embeddings[query_idx].reshape(1, -1)
        dense_sim = cosine_similarity(query_dense, movie_embeddings).flatten()

        # 3. Blending Formula
        hybrid_sim = beta * sparse_sim + (1.0 - beta) * dense_sim
        similar_indices = np.argsort(hybrid_sim)[::-1][1:top_k + 1]

        end_perf = time.perf_counter()
        st.markdown(
            f"⏱️ **Hồ sơ hiệu năng (System Profile):** Trích xuất Dense & Sparse Vector hoàn thành trong **{(end_perf - start_perf) * 1000:.2f} ms**.")

        cols = st.columns(5)
        for i, idx in enumerate(similar_indices):
            movie_info = movies_df.iloc[idx]
            title = movie_info['title']
            genres_raw = movie_info['genres'].split('|')
            poster_url = get_movie_poster(title)

            with cols[i % 5]:
                st.image(poster_url, use_container_width=True)
                badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                st.markdown(f"""
                    <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                    <div>{badges}</div>
                    <span style="font-size:12px; color:#00d2d3; font-weight:bold;">🧬 TƯƠNG ĐỒNG LAI: {hybrid_sim[idx] * 100:.1f}%</span>
                """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ 3: TÌM KIẾM THEO TẬP HỢP THỂ LOẠI (GENRE QUERY ENGINE)
# ==============================================================================
elif app_mode == "🏷️ Tìm Phim Theo Thể Loại":
    st.header("🏷️ Genre Query Engine - Tìm kiếm phim theo tổ hợp danh mục")
    st.markdown(
        "Chọn một hoặc nhiều thể loại kết hợp. Hệ thống sẽ lọc ra các tập hợp dữ liệu tương ứng theo cấu trúc logic.")

    target_genres = st.multiselect("Nhấp chọn các thể loại phim muốn tìm kiếm:", all_genres, default=[all_genres[0]])

    match_type = st.radio(
        "Điều kiện lọc thể loại:",
        ["Chỉ cần chứa ít nhất một dòng phim đã chọn (Logic HOẶC)",
         "Bắt buộc phải chứa toàn bộ các dòng phim đã chọn (Logic VÀ)"],
        horizontal=True
    )

    if st.button("🔍 TIẾN HÀNH TRUY VẤN WAREHOUSE"):
        if not target_genres:
            st.error("Vui lòng chọn ít nhất một thể loại để truy vấn!")
        else:
            with st.spinner('Đang truy vấn kho dữ liệu phim...'):
                if "ít nhất một" in match_type:
                    mask = movies_df['genres'].apply(lambda x: any(g in x.split('|') for g in target_genres))
                else:
                    mask = movies_df['genres'].apply(lambda x: all(g in x.split('|') for g in target_genres))

                filtered_movies = movies_df[mask]

                if len(filtered_movies) == 0:
                    st.warning("⚠️ Không tìm thấy bộ phim nào chứa đúng tổ hợp cấu trúc thể loại này.")
                else:
                    st.success(
                        f"Tìm thấy tổng cộng {len(filtered_movies)} phim. Hiển thị top {min(top_k, len(filtered_movies))} kết quả:")

                    display_df = filtered_movies.head(top_k)
                    cols = st.columns(5)
                    for i, row in display_df.reset_index().iterrows():
                        title = row['title']
                        genres_raw = row['genres'].split('|')
                        poster_url = get_movie_poster(title)

                        with cols[i % 5]:
                            st.image(poster_url, use_container_width=True)
                            badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                            st.markdown(f"""
                                <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                                <div>{badges}</div>
                                <span style="font-size:12px; color:#ff9f43; font-weight:bold;">🎬 INDEX ID: #{row['movie_idx']}</span>
                            """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ 4: PHÒNG THÍ NGHIỆM HỌC MÁY (PCA EMBEDDING PROJECTION & DYNAMIC GRAPHS)
# ==============================================================================
else:
    st.header("🔬 Phòng thí nghiệm tối ưu hóa Gradient Descent & Convergence")
    st.markdown(
        "💡 **Công nghệ Độc quyền:** Đo lường độ dời hướng vị trí không gian Euclidean ($\Delta d$) và giám sát biểu đồ hàm Loss động trong thời gian thực.")

    col_rate1, col_rate2 = st.columns([2, 1])
    with col_rate1:
        chosen_movie = st.selectbox("Chọn bộ phim tương tác gán nhãn:", movies_df['title'].values)
    with col_rate2:
        rating = st.slider("Chấm điểm (Sao):", 1, 5, 5)

    if st.button("➕ Ghi nhận điểm số cục bộ"):
        m_idx = movies_df[movies_df['title'] == chosen_movie].iloc[0]['movie_idx']
        st.session_state.user_ratings[int(m_idx)] = float(rating) / 5.0
        st.success(f"Đã lưu phản hồi vào Session: {chosen_movie} -> {rating}⭐")

    if st.session_state.user_ratings:
        st.markdown("### 📋 Danh sách tập mẫu thử nghiệm (Training Dataset):")
        for m_idx, score in st.session_state.user_ratings.items():
            m_title = movies_df[movies_df['movie_idx'] == m_idx].iloc[0]['title']
            st.markdown(f"`Movie ID #{m_idx:05d}` — {m_title} — Đã chấm: **{score * 5:.0f}⭐**")

        st.markdown("---")
        st.subheader("⚙️ Cấu hình cấu trúc siêu tham số (Hyperparameters)")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            opt_choice = st.selectbox("Thuật toán tối ưu (Optimizer):", ["Adam", "AdamW", "SGD"])
        with c_opt2:
            lr_val = st.number_input("Hệ số học tập (Learning Rate):", min_value=0.0001, max_value=0.1, value=0.01,
                                     format="%.4f")
        with c_opt3:
            w_decay = st.number_input("Hệ số chống quá khớp (Weight Decay):", min_value=0.0, max_value=0.01, value=1e-4,
                                      format="%.5f")

        if st.button("🔥 KÍCH HOẠT QUÁ TRÌNH BACKPROPAGATION"):
            target_user = 80000

            movie_ids = list(st.session_state.user_ratings.keys())
            target_scores = list(st.session_state.user_ratings.values())

            user_t = torch.tensor([target_user] * len(movie_ids), dtype=torch.long).to(device)
            movie_t = torch.tensor(movie_ids, dtype=torch.long).to(device)
            y_true = torch.tensor(target_scores, dtype=torch.float32).to(device)

            # ĐO LƯỜNG TRƯỚC HUẤN LUYỆN (Embedding vector của phim đang tương tác)
            target_movie_idx = int(movies_df[movies_df['title'] == chosen_movie].iloc[0]['movie_idx'])
            t_movie_tensor = torch.tensor([target_movie_idx], dtype=torch.long).to(device)

            model.eval()
            with torch.no_grad():
                emb_before = model.movie_embedding(t_movie_tensor).clone().cpu().numpy().flatten()

            # Cấu hình Optimizer động
            model.train()
            if opt_choice == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr_val, weight_decay=w_decay)
            elif opt_choice == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr_val, weight_decay=w_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr_val, momentum=0.9, weight_decay=w_decay)

            criterion = nn.BCELoss()

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Biểu đồ Loss động
            loss_history = []
            chart_placeholder = st.empty()

            initial_loss = None
            final_loss = None

            # Vòng lặp tối ưu 10 epochs
            for epoch in range(1, 11):
                optimizer.zero_grad()
                predictions = model(user_t, movie_t)

                predictions = predictions.view(-1)
                y_true = y_true.view(-1)

                loss = criterion(predictions, y_true)
                if epoch == 1: initial_loss = loss.item()
                final_loss = loss.item()

                loss_history.append(loss.item())

                # Vẽ trực tiếp biểu đồ biến thiên độ lỗi
                chart_placeholder.line_chart(pd.DataFrame({"Loss": loss_history}))

                loss.backward()
                optimizer.step()

                progress_bar.progress(epoch * 10)
                status_text.text(f"🚀 Thuật toán {opt_choice} đang học... Vòng {epoch}/10 | Loss: {loss.item():.5f}")
                time.sleep(0.05)

            st.balloons()

            # ĐO LƯỜNG SAU HUẤN LUYỆN (Độ dời Euclidean trong không gian vector)
            model.eval()
            with torch.no_grad():
                emb_after = model.movie_embedding(t_movie_tensor).clone().cpu().numpy().flatten()

            # Công thức khoảng cách Euclidean 32 chiều
            vector_drift = np.linalg.norm(emb_after - emb_before)

            # BÁO CÁO KẾT QUẢ
            st.subheader("📊 Báo cáo kiểm định sự hội tụ toán học:")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(label="Độ lỗi khởi tạo (Initial Loss)", value=f"{initial_loss:.4f}")
            with c2:
                delta_loss = final_loss - initial_loss
                st.metric(label="Độ lỗi sau hội tụ (Final Loss)", value=f"{final_loss:.4f}", delta=f"{delta_loss:.4f}")
            with c3:
                convergence_rate = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
                st.metric(label="Tỷ lệ hội tụ (%)", value=f"{convergence_rate:.1f}%")
            with c4:
                st.metric(label="Độ dời vector nhúng (Δd)", value=f"{vector_drift:.6f}")
                st.caption("Khoảng cách vector dịch chuyển trong không gian 32 chiều sau tối ưu")

            # 🆕 TÍNH NĂNG ĐỘC QUYỀN: BẢN ĐỒ KHÔNG GIAN VECTOR 2D (PCA PROJECTION)
            st.markdown("---")
            st.subheader("🗺️ Bản đồ Không không gian Vector Nhúng 2D (PCA Latent Space)")
            st.markdown(
                "💡 **Cơ sở khoa học:** Áp dụng phương thức phân tích thành phần chính (PCA) để nén toàn bộ không gian hành vi 32 chiều xuống không gian 2 chiều trực quan, thể hiện rõ các bộ phim bạn vừa chấm điểm nằm ở đâu trong cụm hành vi.")

            with st.spinner('Đang tính toán ma trận PCA...'):
                rated_movie_idxs = list(st.session_state.user_ratings.keys())
                # Lấy ngẫu nhiên thêm 40 phim khác trong hệ thống để làm nền tham chiếu
                np.random.seed(42)
                random_idxs = list(np.random.choice(10524, size=40, replace=False))
                all_sampled_idxs = list(set(rated_movie_idxs + random_idxs))

                # Trích xuất trọng số vector nhúng hiện tại từ model PyTorch
                model.eval()
                with torch.no_grad():
                    all_sampled_tensor = torch.tensor(all_sampled_idxs, dtype=torch.long).to(device)
                    sampled_embeddings = model.movie_embedding(all_sampled_tensor).cpu().numpy()

                # Thực thi giảm chiều dữ liệu PCA từ 32D -> 2D
                pca = PCA(n_components=2)
                embs_2d = pca.fit_transform(sampled_embeddings)

                # Chuẩn bị DataFrame để vẽ đồ thị
                plot_data = []
                for idx_in_list, movie_idx in enumerate(all_sampled_idxs):
                    m_title = movies_df[movies_df['movie_idx'] == movie_idx].iloc[0]['title']
                    is_rated = "Phim bạn đã chấm" if movie_idx in rated_movie_idxs else "Phim nền tham chiếu"
                    plot_data.append({
                        "Component 1": embs_2d[idx_in_list, 0],
                        "Component 2": embs_2d[idx_in_list, 1],
                        "Tên phim": m_title,
                        "Loại": is_rated
                    })

                plot_df = pd.DataFrame(plot_data)

                # Hiển thị đồ thị phân cụm động
                st.scatter_chart(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    color="Loại",
                    size=15
                )