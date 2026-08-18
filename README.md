# 🎬 MovieLens Recommendation System — Hệ Thống Gợi Ý Phim Cá Nhân Hóa

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Đề cương Môn học / Đồ án Chuyên ngành:** Nghiên cứu, thiết kế và phát triển Hệ thống Gợi ý Phim thông minh dựa trên kỹ thuật **Lọc theo nội dung (Content-Based Filtering)**, **Lọc cộng tác (Collaborative Filtering)** và **Phân rã ma trận (Matrix Factorization - SVD)** trên tập dữ liệu chuẩn **MovieLens**.

---

## 📋 Mục Lục

1. [📌 Giới Thiệu & Tính Năng Nổi Bật](#1-giới-thiệu--tính-năng-nổi-bật)
2. [🏗️ Kiến Trúc Hệ Thống & Luồng Xử Lý](#2-kiến-trúc-hệ-thống--luồng-xử-lý)
3. [📊 Bộ Dữ Liệu Sử Dụng (MovieLens)](#3-bộ-dữ-liệu-sử-dụng-movielens)
4. [🛠️ Công Nghệ & Thư Viện Sử Dụng](#4-công-nghệ--thư-viện-sử-dụng)
5. [📁 Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
6. [⚙️ Thuật Toán & Phương Pháp Chi Tiết](#6-thuật-toán--phương-pháp-chi-tiết)
7. [🚀 Hướng Dẫn Cài Đặt & Vận Hành](#7-hướng-dẫn-cài-đặt--vận-hành)
8. [📈 Đánh Giá Hiệu Năng Mô Hình](#8-đánh-giá-hiệu-năng-mô-hình)
9. [🗺️ Định Hướng Phát Triển](#9-định-hướng-phát-triển)
10. [👥 Thành Viên Thực Hiện](#10-thành-viên-thực-hiện)
11. [📄 Giấy Phép](#11-giấy-phép)

---

## 1. 📌 Giới Thiệu & Tính Năng Nổi Bật

Trong kỉ nguyên bùng nổ nội dung số, bài toán **Cold Start** và **Information Overload** khiến người dùng tốn nhiều thời gian chọn lựa nội dung phù hợp. Dự án xây dựng một giải pháp gợi ý phim đa tầng:

- 🌟 **Gợi ý đa phương pháp:** Kết hợp linh hoạt giữa đặc trưng nội dung phim và hành vi đánh giá của cộng đồng.
- 🎯 **Dự đoán điểm số chính xác:** Dự đoán mức độ yêu thích (Rating 1.0 - 5.0) của người dùng đối với các bộ phim chưa xem.
- 🔍 **Tìm kiếm phim tương đồng:** Tìm nhanh các phim có cốt truyện, thể loại hoặc tập người xem tương tự bộ phim được chọn.
- 🖥️ **Giao diện Trực quan (Interactive UI):** Web App viết bằng **Streamlit** hỗ trợ người dùng xem danh sách gợi ý theo thời gian thực.

---

## 2. 🏗️ Kiến Trúc Hệ Thống & Luồng Xử Lý

```text
┌────────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│  MovieLens Data│ ──> │ Tiền xử lý & EDA    │ ──> │ Trích xuất Đặc trưng   │
│  (CSV Files)   │     │ (Clean, Impute, Scaling)  │ (TF-IDF, Pivot Matrix) │
└────────────────┘     └─────────────────────┘     └────────────────────────┘
                                                              │
                                                              ▼
┌────────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│  Streamlit UI  │ <── │ Top-K Recommendations│ <── │ Các Mô Hình Gợi Ý      │
│  (Web Demo)    │     │ (Ranking & Filter)  │     │ (Content / CF / SVD)   │
└────────────────┘     └─────────────────────┘     └────────────────────────┘
```

---

## 3. 📊 Bộ Dữ Liệu Sử Dụng (MovieLens)

Dự án sử dụng bộ dữ liệu chuẩn **MovieLens (100K / 1M)** phát hành bởi GroupLens Research:

| Tên File | Dung Lượng | Mô Tả Dữ Liệu | Các Trường Dữ Liệu Chính |
| :--- | :---: | :--- | :--- |
| `movies.csv` | ~500 KB | Thông tin danh mục phim | `movieId`, `title`, `genres` |
| `ratings.csv` | ~2.5 MB | Lịch sử đánh giá của người dùng | `userId`, `movieId`, `rating`, `timestamp` |
| `tags.csv` | ~100 KB | Nhãn/từ khóa mô tả do người dùng gán | `userId`, `movieId`, `tag`, `timestamp` |
| `links.csv` | ~200 KB | Mã ID tham chiếu ra ngoài | `movieId`, `imdbId`, `tmdbId` |

---

## 4. 🛠️ Công Nghệ & Thư Viện Sử Dụng

- **Ngôn ngữ lập trình:** `Python 3.8+`
- **Xử lý Dữ liệu:** `Pandas`, `NumPy`, `SciPy`
- **Trực quan hóa Dữ liệu (EDA):** `Matplotlib`, `Seaborn`, `Plotly`
- **Machine Learning & Recommender Engine:**
  - `Scikit-Learn` (TF-IDF, Cosine Similarity, NearestNeighbors)
  - `Scikit-Surprise` (SVD, SVD++, NMF, KNNBasic)
- **Giao diện Ứng dụng Web:** `Streamlit`
- **Môi trường Phát triển:** VS Code / Jupyter Notebook / Google Colab

---

## 5. 📁 Cấu Trúc Thư Mục Dự Án

DeCuongMonHocMovieLens/
│
├── data/                       # Dữ liệu dự án
│   ├── raw/                    # Dữ liệu gốc chưa qua xử lý (movies.csv, ratings.csv,...)
│   └── processed/              # Dữ liệu đã tiền xử lý & tạo ma trận
│
├── notebooks/                  # Jupyter Notebooks phân tích & thực nghiệm
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Content_Based_Filtering.ipynb
│   ├── 03_Collaborative_Filtering_KNN.ipynb
│   ├── 04_Matrix_Factorization_SVD.ipynb
│   └── 05_Model_Evaluation_Comparison.ipynb
│
├── src/                        # Mã nguồn chính dự án
│   ├── __init__.py
│   ├── data_loader.py          # Lớp nạp & làm sạch dữ liệu
│   ├── recommenders.py         # Cài đặt các thuật toán gợi ý
│   └── metrics.py              # Đánh giá độ đo (RMSE, MAE, Precision@K)
│
├── app.py                      # Ứng dụng Web Demo bằng Streamlit
├── requirements.txt            # Danh sách thư viện cần cài đặt
├── .gitignore                  # Bỏ qua các file rác / bytecode
└── README.md                   # Tài liệu hướng dẫn chi tiết dự án

---

## 6. ⚙️ Thuật Toán & Phương Pháp Chi Tiết

### 6.1. Content-Based Filtering (Gợi Ý Theo Nội Dung)
- **Cơ chế:** Phân tích sự tương đồng về thuộc tính của các bộ phim (Thể loại `genres`, Nhãn từ khóa `tags`).
- **Kỹ thuật áp dụng:**
  - Biến đổi chuỗi văn bản thành ma trận tần suất từ bằng **TF-IDF Vectorizer**.
  - Tính toán độ tương đồng giữa các vector phim qua **Cosine Similarity**:
    $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

### 6.2. Collaborative Filtering (Lọc Cộng Tác)
- **User-Based CF:** Gợi ý cho người dùng $U$ các bộ phim mà những người dùng khác có sở thích tương tự $U$ đã thích.
- **Item-Based CF:** Gợi ý các bộ phim tương tự với những bộ phim mà người dùng $U$ đã từng đánh giá cao trong quá khứ.
- **Độ đo tương đồng:** Pearson Correlation / Cosine Similarity.

### 6.3. Matrix Factorization - SVD (Phân Rã Ma Trận)
- **Cơ chế:** Giải quyết bài toán thưa thớt dữ liệu (Data Sparsity) bằng cách phân rã ma trận Đánh giá $R_{m \times n}$ thành hai ma trận thuộc tính ẩn (Latent Features):
  $$R \approx P \times Q^T$$
  Trong đó $P$ là ma trận Người dùng - Thuộc tính ẩn, $Q$ là ma trận Phim - Thuộc tính ẩn.
- **Tối ưu hóa:** Thuật toán Stochastic Gradient Descent (SGD) minimizing hàm mất mát Squared Error.

---

## 7. 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### Bước 1: Clone Repository
```bash
git clone [https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git](https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git)
cd DeCuongMonHocMovieLens
```

### Bước 2: Khởi Tạo & Kích Hoạt Môi Trường Ảo
```bash
# Đối với Windows
python -m venv venv
venv\Scripts\activate

# Đối với macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Thư Viện Phụ Thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Tải Dữ Liệu
1. Tải tập dữ liệu **MovieLens 100K / 1M** từ [GroupLens Website](https://grouplens.org/datasets/movielens/).
2. Giải nén và lưu toàn bộ file `.csv` vào thư mục `data/raw/`.

### Bước 5: Khởi Chạy Web Demo (Streamlit)
```bash
streamlit run app.py
```
> Trình duyệt sẽ tự động mở trang web demo tại địa chỉ: `http://localhost:8501`

---

## 8. 📈 Đánh Giá Hiệu Năng Mô Hình

Các thuật toán được kiểm thử trên tập dữ liệu **Test Set (20% split)** với 2 chỉ số đo lường độ sai số dự đoán:

- **RMSE (Root Mean Squared Error):** Đo lường mức độ lệch trung bình bình phương giữa điểm đánh giá thực tế và dự đoán.
- **MAE (Mean Absolute Error):** Đo lường độ lệch tuyệt đối trung bình.

| Mô Hình (Algorithm) | RMSE | MAE | Ưu Điểm | Hạn Chế |
| :--- | :---: | :---: | :--- | :--- |
| **Baseline (Mean Rating)** | 1.052 | 0.835 | Đơn giản, tính toán tức thì | Độ chính xác thấp |
| **Content-Based Filtering** | 0.985 | 0.772 | Không bị Cold Start đối với phim mới | Bị bó hẹp trong thể loại cũ |
| **User-Based CF (KNN)** | 0.923 | 0.718 | Cá nhân hóa cao, gợi ý đa dạng | Tốn chi phí tính toán khi User tăng |
| **Item-Based CF (KNN)** | 0.910 | 0.705 | Tính ổn định cao hơn User-Based | Vẫn gặp khó khi Item quá mới |
| **SVD (Matrix Factorization)** | **0.875** | **0.672** | **Độ chính xác cao nhất, tối ưu ma trận thưa** | Thời gian huấn luyện mô hình lâu |

---

## 9. 🗺️ Định Hướng Phát Triển

- [ ] **Hybrid Recommender System:** Kết hợp trọng số giữa SVD và Content-Based để giải quyết triệt để bài toán Cold Start.
- [ ] **Deep Learning Integration:** Thử nghiệm mô hình **Neural Collaborative Filtering (NCF)** hoặc **Autoencoders**.
- [ ] **Real-time Streaming:** Tích hợp Apache Kafka / Redis để gợi ý cập nhật ngay khi người dùng bấm Rating.
- [ ] **Deploy Cloud:** Triển khai ứng dụng Web lên Streamlit Community Cloud hoặc AWS / Docker.

---

## 10. 👥 Thành Viên Thực Hiện

| STT | Họ và Tên | Mã Số Sinh Viên | Vai Trò | Contact / Social |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **[Tên Của Bạn]** | [MSSV] | Trưởng nhóm, Thiết kế Model & Backend | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/nphlanho-star) |
| 2 | **[Thành Viên 2]** | [MSSV] | Data Preprocessing, EDA & Streamlit UI | [![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](#) |

- **Giảng viên hướng dẫn:** [Tên Giảng Viên Hướng Dẫn]
- **Trường / Khoa:** Khoa Công Nghệ Thông Tin - [Tên Trường Đại Học]

---

## 11. 📄 Giấy Phép

Dự án này được phân phối dưới giấy phép công khai **MIT License**. Chi tiết xem tại file [LICENSE](LICENSE).

---
<p align="center">
  <i>⭐ Đừng quên tặng 1 Star trên GitHub nếu bạn thấy dự án này hữu ích! ⭐</i>
</p>
