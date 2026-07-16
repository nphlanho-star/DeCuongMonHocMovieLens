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

# ==============================================================================
# CẤU HÌNH TỰ ĐỘNG KÍCH HOẠT HẠ TẦNG STREAMLIT
# ==============================================================================
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    if os.environ.get("IS_STREAMLIT_RUNNING") != "1":
        os.environ["IS_STREAMLIT_RUNNING"] = "1"
        print("🚀 Đang kích hoạt hạ tầng tính toán song song...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit()

# ==============================================================================
# QUẢN LÝ PHẦN CỨNG VẬT LÝ (HARDWARE ACCELERATION ORCHESTRATOR)
# ==============================================================================
# Tự động tối ưu phần cứng: Ưu tiên CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# GIAO DIỆN CINEMATIC PREMIUM & HIỆU ỨNG CARD 3D
# ==============================================================================
st.set_page_config(page_title="Netflix Enterprise v4.0", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #060608; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #888888; background-color: transparent; font-weight: bold; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #E50914 !important; border-bottom-color: #E50914 !important; }

    /* Thiết kế Glassmorphism Card đổ bóng 3D */
    div[data-testid="stBlock"] { 
        background: linear-gradient(145deg, #101014, #171721); 
        padding: 16px; border-radius: 14px; border: 1px solid #20202a; 
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1); margin-bottom: 25px; 
        box-shadow: 0 8px 24px rgba(0,0,0,0.6); 
    }
    div[data-testid="stBlock"]:hover { 
        transform: translateY(-6px) scale(1.01); border-color: #E50914; 
        box-shadow: 0 14px 28px rgba(229,9,20,0.25); 
    }
    .genre-tag { display: inline-block; background-color: #1e1e28; color: #ff4757; padding: 2px 10px; border-radius: 20px; font-size: 10px; margin-right: 4px; margin-bottom: 4px; border: 1px solid #2d2d3d; font-weight: bold; }
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
            nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, user_indices, movie_indices):
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        x = torch.cat([user_embed, movie_embed], dim=-1)
        return self.fc_layers(x).squeeze()


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
    model = model.to(device)  # Đồng bộ mô hình lên chip xử lý tăng tốc
    movies_df = pd.read_csv('movies_mapped.csv')

    # Khởi tạo ma trận không gian đặc trưng NLP
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].str.replace('|', ' ') + ' ' + movies_df['title'])

    return model, movies_df, tfidf_matrix


model, movies_df, tfidf_matrix = load_resources()
all_genres = sorted(list(set([g for g_list in movies_df['genres'].str.split('|') for g in g_list])))

# --- DASHBOARD CONTROL PANEL ---
st.title("🎬 NETFLIX ENTERPRISE SYSTEMS - V4.0 PLATFORM SUPER ENGINE")
st.caption(
    "Kiến trúc công nghiệp: Thực thi đa phần cứng (Hardware-Agnostic) song hành cùng Hệ thống đo lường hiệu năng chuyên sâu")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1: st.metric(label="🖥️ Thiết bị tính toán hiện tại", value=str(device).upper())
with m_col2: st.metric(label="📊 Chiều Vector Embedding", value="32-Dimensions")
with m_col3: st.metric(label="⚙️ Động cơ tối ưu mạng AI", value="Adam Optimizer")
with m_col4: st.metric(label="📈 Tiêu chuẩn tính toán lỗi", value="Binary Cross Entropy")

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
# PHÂN HỆ 1: CÁ NHÂN HÓA QUA MẠNG NEURAL (NCF DEEP LEARNING)
# ==============================================================================
if app_mode == "👤 Cá nhân hóa AI (NCF)":
    st.header("🎯 Gợi ý cá nhân hóa Deep Learning (NCF)")

    col_input1, col_input2 = st.columns([2, 2])
    with col_input1:
        user_id_input = st.number_input("Nhập mã định danh User ID (0 - 85306):", min_value=0, max_value=85306,
                                        value=80000)
    with col_input2:
        selected_genre = st.selectbox("Lọc nhanh theo dòng phim:", ["Tất cả thể loại"] + all_genres)

    if st.button("🚀 TRUY XUẤT VECTOR HÀNH VI"):
        with st.spinner('Mạng Neural đang dự đoán song song...'):
            start_perf = time.perf_counter()  # Bắt đầu đo thời gian xử lý mạng Neural

            model.eval()
            user_tensor = torch.tensor([user_id_input] * 10524, dtype=torch.long).to(device)
            movie_tensor = torch.tensor(list(range(10524)), dtype=torch.long).to(device)

            with torch.no_grad():
                scores = model(user_tensor, movie_tensor).cpu().numpy()

            all_indices = np.argsort(scores)[::-1]
            valid_indices, valid_scores = [], []

            for idx in all_indices:
                movie_info = movies_df[movies_df['movie_idx'] == idx].iloc[0]
                if selected_genre == "Tất cả thể loại" or selected_genre in movie_info['genres'].split('|'):
                    valid_indices.append(idx)
                    valid_scores.append(scores[idx])
                if len(valid_indices) == top_k: break

            end_perf = time.perf_counter()
            st.markdown(
                f"⏱️ **Hồ sơ hiệu năng (System Profile):** Mạng Neural thực hiện Forward-pass mất **{(end_perf - start_perf) * 1000:.2f} ms** trên phần cứng **{str(device).upper()}**.")

            if not valid_indices:
                st.warning("Không tìm thấy bộ phim nào phù hợp.")
            else:
                cols = st.columns(5)
                for i, idx in enumerate(valid_indices):
                    movie_info = movies_df[movies_df['movie_idx'] == idx].iloc[0]
                    title = movie_info['title']
                    genres_raw = movie_info['genres'].split('|')
                    poster_url = get_movie_poster(title)

                    with cols[i % 5]:
                        st.image(poster_url, use_container_width=True)
                        badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_raw])
                        st.markdown(f"""
                            <h5 style="margin:5px 0; font-size:13px; font-weight:bold; height:35px; overflow:hidden;">{title}</h5>
                            <div>{badges}</div>
                            <span style="font-size:12px; color:#E50914; font-weight:bold;">🔥 PHÙ HỢP: {valid_scores[i] * 100:.1f}%</span>
                        """, unsafe_allow_html=True)

# ==============================================================================
# PHÂN HỆ 2: TÌM KIẾM PHIM THEO TÊN/TỪ KHÓA (NLP CONTENT)
# ==============================================================================
elif app_mode == "🔍 Tìm Phim Qua Từ Khóa/Tên (NLP)":
    st.header("🔍 NLP Search Engine - Phân tích độ tương đồng nội dung văn bản")

    search_query = st.selectbox("Chọn bộ phim gốc làm tâm điểm dữ liệu:", movies_df['title'].values)

    if st.button("🔍 PHÂN TÍCH ĐẶC TRƯNG CỐT TRUYỆN"):
        start_perf = time.perf_counter()
        query_idx = movies_df[movies_df['title'] == search_query].index[0]

        # Tính khoảng cách không gian Cosine dựa trên TF-IDF matrix
        cosine_sim = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()
        similar_indices = np.argsort(cosine_sim)[::-1][1:top_k + 1]
        end_perf = time.perf_counter()

        st.markdown(
            f"⏱️ **Hồ sơ hiệu năng (System Profile):** Máy học trích xuất ma trận Cosine mất **{(end_perf - start_perf) * 1000:.2f} ms**.")

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
                    <span style="font-size:12px; color:#00d2d3; font-weight:bold;">🧬 TƯƠNG ĐỒNG: {cosine_sim[idx] * 100:.1f}%</span>
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
# PHÂN HỆ 4: PHÒNG THÍ NGHIỆM HỌC MÁY CỤC BỘ (ACTIVE LEARNING - SỬA TRIỆT ĐỂ LỖI DIM)
# ==============================================================================
else:
    st.header("🔬 Model Fine-Tuning & Dynamic Convergence Metrics Profile")
    st.info(
        "💡 Điểm cốt lõi: Bạn tương tác gán nhãn $\rightarrow$ Ép mạng Neural PyTorch học và bẻ cong Embedding Vector trên chip phần cứng ngay tại chỗ.")

    col_rate1, col_rate2 = st.columns([2, 1])
    with col_rate1:
        chosen_movie = st.selectbox("Chọn bộ phim bạn đã xem để chấm điểm:", movies_df['title'].values)
    with col_rate2:
        rating = st.slider("Chấm điểm (Sao):", 1, 5, 5)

    if st.button("➕ Ghi nhận điểm số cục bộ"):
        m_idx = movies_df[movies_df['title'] == chosen_movie].iloc[0]['movie_idx']
        st.session_state.user_ratings[int(m_idx)] = float(
            rating) / 5.0  # Chuẩn hóa về đoạn [0,1] để khớp hàm Sigmoid đầu ra
        st.success(f"Đã lưu phản hồi vào Session: {chosen_movie} -> {rating}⭐")

    if st.session_state.user_ratings:
        st.markdown("### 📋 Danh sách tập mẫu thử nghiệm (Training Dataset):")
        for m_idx, score in st.session_state.user_ratings.items():
            m_title = movies_df[movies_df['movie_idx'] == m_idx].iloc[0]['title']
            st.markdown(f"`Movie ID #{m_idx:05d}` — {m_title} — Đã chấm: **{score * 5:.0f}⭐**")

        st.markdown("---")
        st.subheader("🧠 Quá trình học máy Lan truyền ngược (Backpropagation)")

        if st.button("🔥 KÍCH HOẠT HUẤN LUYỆN ON-DEVICE"):
            target_user = 80000

            movie_ids = list(st.session_state.user_ratings.keys())
            target_scores = list(st.session_state.user_ratings.values())

            # Khởi tạo Tensor dữ liệu và đẩy thẳng lên thiết bị tính toán tối ưu (CPU/CUDA/MPS)
            user_t = torch.tensor([target_user] * len(movie_ids), dtype=torch.long).to(device)
            movie_t = torch.tensor(movie_ids, dtype=torch.long).to(device)
            y_true = torch.tensor(target_scores, dtype=torch.float32).to(device)

            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()

            progress_bar = st.progress(0)
            status_text = st.empty()

            initial_loss = None
            final_loss = None

            # Vòng lặp lan truyền ngược 10 Epochs
            for epoch in range(1, 11):
                optimizer.zero_grad()
                predictions = model(user_t, movie_t)

                # CHUẨN HÓA KÍCH THƯỚC: Ép phẳng mảng tránh lỗi mismatch kể cả khi chỉ có 1 phần tử
                predictions = predictions.view(-1)
                y_true = y_true.view(-1)

                loss = criterion(predictions, y_true)
                if epoch == 1: initial_loss = loss.item()
                final_loss = loss.item()

                loss.backward()
                optimizer.step()

                progress_bar.progress(epoch * 10)
                status_text.text(
                    f"🚀 Thuật toán Gradient Descent đang tối ưu... Vòng {epoch}/10 | Loss: {loss.item():.5f}")
                time.sleep(0.04)  # Giãn cách nhỏ giúp giao diện mượt mà trực quan

            st.balloons()

            # Xuất dữ liệu kiểm tra toán học phục vụ làm báo cáo, slide thuyết trình đồ án
            st.subheader("📊 Báo cáo kiểm định sự hội tụ toán học:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Độ lỗi khởi tạo (Initial Loss)", value=f"{initial_loss:.4f}")
            with c2:
                delta_loss = final_loss - initial_loss
                st.metric(label="Độ lỗi sau hội tụ (Final Loss)", value=f"{final_loss:.4f}", delta=f"{delta_loss:.4f}")
            with c3:
                convergence_rate = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
                st.metric(label="Tỷ lệ hội tụ thuật toán (%)", value=f"{convergence_rate:.1f}%")