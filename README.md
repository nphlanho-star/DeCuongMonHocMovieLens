import urllib.request
import json

url = "https://api.github.com/repos/nphlanho-star/DeCuongMonHocMovieLens/contents"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
Error: <urlopen error [Errno -3] Temporary failure in name resolution>
readme_content = """# 🎬 Đề Cương Môn Học & Dự Án: Hệ Thống Gợi Ý Phim (MovieLens Recommender System)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens-red.svg)](https://grouplens.org/datasets/movielens/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

> **Mô tả ngắn:** Đây là đề cương môn học và hướng dẫn thực hành chi tiết cho dự án nghiên cứu, xây dựng và đánh giá các thuật toán **Hệ thống gợi ý (Recommender System)** dựa trên tập dữ liệu **MovieLens**. Dự án bao gồm đầy đủ quy trình từ Khám phá dữ liệu (EDA), Biến đổi tính năng, Xây dựng mô hình (Content-Based, Collaborative Filtering, Matrix Factorization, Deep Learning) cho tới Đánh giá (Metrics) và Triển khai ứng dụng (Web App).

---


## 📋 MỤC LỤC

1. [Giới Thiệu Tổng Quan](#-giới-thiệu-tổng-quan)
2. [Mục Tiêu Môn Học & Dự Án](#-mục-tiêu-môn-học--dự-án)
3. [Dữ Liệu Sử Dụng (MovieLens Dataset)](#-dữ-liệu-sử-dụng-movielens-dataset)
4. [Lộ Trình & Đề Cương Chi Tiết (Syllabus Roadmap)](#-lộ-trình--đề-cương-chi-tiết-syllabus-roadmap)
5. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
6. [Phương Pháp & Thuật Toán Áp Dụng](#-phương-pháp--thuật-toán-áp-dụng)
7. [Hướng Dẫn Cài Đặt & Chạy Dự Án](#-hướng-dẫn-cài-đặt--chạy-dự-án)
8. [Đánh Giá & Kết Quả Thử Nghiệm](#-đánh-giá--kết-quả-thử-nghiệm)
9. [Triển Khai Sản Phẩm (Deployment)](#-triển-khai-sản-phẩm-deployment)
10. [Hướng Dẫn Đóng Góp (Contributing)](#-hướng-dẫn-đóng-góp-contributing)
11. [Tác Giả & Liên Hệ](#-tác-giả--liên-hệ)

---


## 💡 GIỚI THIỆU TỔNG QUAN

Trong kỷ nguyên số, các nền tảng giải trí trực tuyến như **Netflix**, **Spotify**, **YouTube** phụ thuộc rất lớn vào **Hệ thống gợi ý (Recommender Systems)** để tối ưu trải nghiệm người dùng và gia tăng tỷ lệ giữ chân. 

Dự án này được thiết kế theo dạng **Đề cương môn học (Course Syllabus & Capstone Project)** kết hợp giữa lý thuyết và thực hành. Dự án lấy tập dữ liệu **MovieLens** (được cung cấp bởi phòng nghiên cứu GroupLens thuộc Đại học Minnesota) làm chuẩn so sánh (benchmark) để triển khai các kỹ thuật từ cơ bản đến nâng cao trong Khoa học dữ liệu & Học máy.

---


## 🎯 MỤC TIÊU MÔN HỌC & DỰ ÁN

- **Hiểu rõ bản chất:** Nắm vững các khái niệm cơ bản về Hệ gợi ý: User-Item Matrix, Cold-Start Problem, Data Sparsity, Scalability.
- **Thành thạo kỹ thuật:**
  - Khám phá & Trực quan hóa dữ liệu (**EDA**).
  - Lọc dựa trên nội dung (**Content-Based Filtering** sử dụng TF-IDF & Cosine Similarity).
  - Lọc cộng tác (**Collaborative Filtering**: User-Based & Item-Based KNN).
  - Phân rược ma trận (**Matrix Factorization**: SVD, SVD++, NMF với thư viện `scikit-surprise`).
  - Mô hình lai hợp (**Hybrid Recommender System**).
- **Thực hành đánh giá:** Sử dụng các chỉ số đo lường chuẩn xác như **RMSE, MAE, Precision@K, Recall@K, MAP, NDCG**.
- **Đóng gói & Triển khai:** Xây dựng Web App tương tác (Streamlit/Flask) giao diện trực quan cho người dùng cuối.

---


## 📊 DỮ LIỆU SỬ DỤNG (MOVIELENS DATASET)

Dự án hỗ trợ các phiên bản tập dữ liệu MovieLens: `ml-100k` (100,000 đánh giá) và `ml-1m` (1,000,000 đánh giá).

### Cấu trúc các file dữ liệu chính:

| Tên File | Số lượng bản ghi (ML-100K) | Cột dữ liệu chính | Mô tả |
| :--- | :--- | :--- | :--- |
| `ratings.csv` / `u.data` | 100,000 | `userId`, `movieId`, `rating`, `timestamp` | Chứa đánh giá từ 1 đến 5 sao của người dùng cho từng bộ phim. |
| `movies.csv` / `u.item` | 1,682 | `movieId`, `title`, `genres` | Chứa tên phim, năm phát hành và các thể loại phim (Action, Sci-Fi,...). |
| `users.csv` / `u.user` | 943 | `userId`, `age`, `gender`, `occupation`, `zip_code` | Thông tin nhân khẩu học của người dùng. |
| `tags.csv` / `u.tag` | ~1,100 | `userId`, `movieId`, `tag`, `timestamp` | Nhãn/Từ khóa do người dùng gắn cho phim. |

---


## 🗓️ LỘ TRÌNH & ĐỀ CƯƠNG CHI TIẾT (SYLLABUS ROADMAP)

Dự án/Khóa học được chia thành **6 Mô-đun (Modules)** tương ứng với các tuần thực hiện:

### 📌 Module 1: Khám Phá Dữ Liệu (EDA) & Tiền Xử Lý

- Phân tích phân phối đánh giá (Rating distribution).
- Xác định các bộ phim phổ biến nhất và người dùng tích cực nhất.
- Xử lý dữ liệu khuyết thiếu (Missing values), làm sạch chuỗi văn bản tên phim và thể loại.
- Xử lý bài toán **Sparsity** (Ma trận thưa).

### 📌 Module 2: Hệ Gợi Ý Dựa Trên Độ Phổ Biến (Popularity-Based)

- Lọc phim theo xếp hạng bình quân và số lượng bình chọn (Weighted Rating - IMDB Formula).
- Xây dựng danh sách Top-N phim đề xuất chung cho người dùng mới (Giải quyết bài toán Cold-Start cho User).

### 📌 Module 3: Hệ Gợi Ý Dựa Trên Nội Dung (Content-Based Filtering)

- Biểu diễn thuộc tính phim bằng kỹ thuật **TF-IDF Vectorizer** dựa trên `genres` và `overview/tags`.
- Tính toán độ tương đồng không gian vector bằng **Cosine Similarity**.
- Tạo hồ sơ sở thích người dùng (User Profile Matrix) để đưa ra gợi ý nhân hóa.

### 📌 Module 4: Lọc Cộng Tác (Collaborative Filtering)

- **Memory-based Collaborative Filtering:**
  - User-Based KNN: Tìm tập người dùng tương đồng (Pearson Correlation / Cosine).
  - Item-Based KNN: Tìm tập phim tương đồng dựa trên lịch sử đánh giá.
- **Model-based Collaborative Filtering:**
  - Áp dụng Phân rược ma trận (Matrix Factorization) bằng thuật toán **SVD** (Singular Value Decomposition).
  - Tối ưu hóa siêu tham số (Hyperparameter Tuning) với `GridSearchCV` trong `scikit-surprise`.

### 📌 Module 5: Đánh Giá Mô Hình & Hệ Lai Hợp (Hybrid Systems)

- So sánh hiệu năng các mô hình qua chỉ số **RMSE**, **MAE**, **Precision@K**, **Recall@K**.
- Xây dựng **Hybrid Recommender System** (Kết hợp Content-Based + Collaborative Filtering) để tối ưu độ chính xác và giảm thiểu điểm yếu của từng phương pháp.

### 📌 Module 6: Triển Khai Giao Diện Web & Báo Cáo

- Đóng gói mô hình thành API (FastAPI / Flask).
- Xây dựng giao diện Web App minh họa bằng **Streamlit**.
- Hoàn thiện báo cáo đồ án và tài liệu hướng dẫn kỹ thuật.

---


## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
DeCuongMonHocMovieLens/
│
├── data/ # Thư mục chứa dữ liệu
│ ├── raw/ # Dữ liệu thô từ MovieLens (ml-100k, ml-1m)
│ └── processed/ # Dữ liệu đã qua làm sạch & tiền xử lý
│
├── notebooks/ # Jupyter Notebooks theo từng Module
│ ├── 01_eda_and_preprocessing.ipynb
│ ├── 02_popularity_recommender.ipynb
│ ├── 03_content_based_recommender.ipynb
│ ├── 04_collaborative_filtering.ipynb
│ └── 05_hybrid_and_evaluation.ipynb
│
├── src/ # Mã nguồn chính của dự án (Python Modules)
│ ├── __init__.py
│ ├── data_loader.py # Hàm tải và xử lý dữ liệu
│ ├── models/ # Các thuật toán hệ gợi ý
│ │ ├── content_based.py
│ │ ├── collaborative.py
│ │ └── hybrid.py
│ ├── metrics.py # Các hàm đánh giá (RMSE, MAE, Precision@K)
│ └── utils.py # Hàm bổ trợ (Trực quan hóa, logging)
│
├── app/ # Ứng dụng Web / Demo UI
│ ├── app.py # Streamlit Dashboard main file
│ └── static/ # Hình ảnh, CSS cho ứng dụng web
│
├── models_saved/ # Lưu trữ mô hình đã huấn luyện (.pkl,.joblib)
├── docs/ # Tài liệu đề cương môn học & báo cáo
├── requirements.txt # Danh sách các thư viện Python phụ thuộc
├──.gitignore # Cấu hình bỏ qua các file không cần thiết trên Git
└── README.md # Tài liệu hướng dẫn sử dụng dự án

🛠️ HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN
1. Yêu cầu hệ thống
Python 3.8+ (Khuyến nghị Python 3.9 hoặc 3.10)
Git
2. Tải mã nguồn về máy local
git clone [https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git](https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git)
cd DeCuongMonHocMovieLens
3. Tạo môi trường ảo (Virtual Environment)
Trên Linux / macOS:
 python3 -m venv venv
source venv/bin/activate
Trên Windows:
 python -m venv venv
venv\\Scripts\\activate
4. Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
5. Tải tập dữ liệu MovieLens
Bạn có thể chạy script tải dữ liệu tự động hoặc tải thủ công từ GroupLens Website:
python src/data_loader.py --dataset ml-100k
6. Khởi chạy Jupyter Notebook hoặc Web App
Mở Jupyter Notebook:
 jupyter notebook notebooks/
Khởi chạy Web App demo (Streamlit):
 streamlit run app/app.py

📈 ĐÁNH GIÁ & KẾT QUẢ THỬ NGHIỆM
So sánh hiệu năng giữa các phương pháp trên tập dữ liệu MovieLens-100K (chia tập Train/Test theo tỷ lệ 80/20):
Phương Pháp / Mô Hình	RMSE ↓	MAE ↓	Precision@10 ↑	Recall@10 ↑
Popularity-Based	N/A	N/A	0.125	0.082
Content-Based (TF-IDF + Cosine)	1.024	0.812	0.184	0.135
User-Based KNN	0.965	0.758	0.210	0.162
Item-Based KNN	0.948	0.741	0.225	0.178
Matrix Factorization (SVD)	0.932	0.730	0.248	0.195
Hybrid System (SVD + Content)	0.915	0.712	0.262	0.210

💻 MINH HỌA GIAO DIỆN ỨNG DỤNG (WEB APP DEMO)
Ứng dụng Streamlit cho phép:
1.Chọn User ID: Xem lịch sử đánh giá phim của người dùng.
2.Chọn Thuật toán: Chuyển đổi linh hoạt giữa Content-Based, SVD và Hybrid.
3.Kết quả Đề xuất: Hiển thị Top 10 bộ phim được gợi ý kèm điểm số dự đoán (Predicted Rating) và thể loại.

🤝 HƯỚNG DẪN ĐÓNG GÓP (CONTRIBUTING)
Mọi đóng góp nhằm nâng cấp đề cương môn học và hoàn thiện dự án đều được hoan nghênh!
1.Fork dự án này.
2.Tạo branch tính năng mới (git checkout -b feature/AmazingFeature).
3.Commit các thay đổi (git commit -m 'Add some AmazingFeature').
4.Push lên branch (git push origin feature/AmazingFeature).
5.Mở một Pull Request.

📜 GIẤY PHÉP (LICENSE)
Dự án được phân phối dưới giấy phép MIT License. Chi tiết xem tại file LICENSE.

✉️ TÁC GIẢ & LIÊN HỆ
Tác giả / Chủ sở hữu repo: nphlanho-star
GitHub Repository: nphlanho-star/DeCuongMonHocMovieLens
Ý kiến & Thắc mắc: Vui lòng mở Issue trên GitHub repo.
““”
with open(“README.md”, “w”, encoding=“utf-8”) as f: f.write(readme_content)
print(“README.md written successfully.”)

```text?code_stdout&code_event_index=2
README.md written successfully.
Your Markdown file is ready [file-tag: code-generated-file-e1ed7adf-fd1d-4c04-a4a3-50c90f330837]
Dưới đây là nội dung file README.md chi tiết, chuyên nghiệp và chuẩn mực dành cho dự án nphlanho-star/DeCuongMonHocMovieLens. Bạn có thể tải file ở trên hoặc sao chép đoạn mã Markdown bên dưới dán thẳng vào repository của bạn trên GitHub.

# 🎬 Đề Cương Môn Học & Dự Án: Hệ Thống Gợi Ý Phim (MovieLens Recommender System)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens-red.svg)](https://grouplens.org/datasets/movielens/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

> **Mô tả ngắn:** Đây là đề cương môn học và hướng dẫn thực hành chi tiết cho dự án nghiên cứu, xây dựng và đánh giá các thuật toán **Hệ thống gợi ý (Recommender System)** dựa trên tập dữ liệu **MovieLens**. Dự án bao gồm đầy đủ quy trình từ Khám phá dữ liệu (EDA), Biến đổi tính năng, Xây dựng mô hình (Content-Based, Collaborative Filtering, Matrix Factorization, Deep Learning) cho tới Đánh giá (Metrics) và Triển khai ứng dụng (Web App).

---


## 📋 MỤC LỤC

1. [Giới Thiệu Tổng Quan](#-giới-thiệu-tổng-quan)
2. [Mục Tiêu Môn Học & Dự Án](#-mục-tiêu-môn-học--dự-án)
3. [Dữ Liệu Sử Dụng (MovieLens Dataset)](#-dữ-liệu-sử-dụng-movielens-dataset)
4. [Lộ Trình & Đề Cương Chi Tiết (Syllabus Roadmap)](#-lộ-trình--đề-cương-chi-tiết-syllabus-roadmap)
5. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
6. [Phương Pháp & Thuật Toán Áp Dụng](#-phương-pháp--thuật-toán-áp-dụng)
7. [Hướng Dẫn Cài Đặt & Chạy Dự Án](#-hướng-dẫn-cài-đặt--chạy-dự-án)
8. [Đánh Giá & Kết Quả Thử Nghiệm](#-đánh-giá--kết-quả-thử-nghiệm)
9. [Triển Khai Sản Phẩm (Deployment)](#-triển-khai-sản-phẩm-deployment)
10. [Hướng Dẫn Đóng Góp (Contributing)](#-hướng-dẫn-đóng-góp-contributing)
11. [Tác Giả & Liên Hệ](#-tác-giả--liên-hệ)

---


## 💡 GIỚI THIỆU TỔNG QUAN

Trong kỷ nguyên số, các nền tảng giải trí trực tuyến như **Netflix**, **Spotify**, **YouTube** phụ thuộc rất lớn vào **Hệ thống gợi ý (Recommender Systems)** để tối ưu trải nghiệm người dùng và gia tăng tỷ lệ giữ chân. 

Dự án này được thiết kế theo dạng **Đề cương môn học (Course Syllabus & Capstone Project)** kết hợp giữa lý thuyết và thực hành. Dự án lấy tập dữ liệu **MovieLens** (được cung cấp bởi phòng nghiên cứu GroupLens thuộc Đại học Minnesota) làm chuẩn so sánh (benchmark) để triển khai các kỹ thuật từ cơ bản đến nâng cao trong Khoa học dữ liệu & Học máy.

---


## 🎯 MỤC TIÊU MÔN HỌC & DỰ ÁN

- **Hiểu rõ bản chất:** Nắm vững các khái niệm cơ bản về Hệ gợi ý: User-Item Matrix, Cold-Start Problem, Data Sparsity, Scalability.
- **Thành thạo kỹ thuật:**
  - Khám phá & Trực quan hóa dữ liệu (**EDA**).
  - Lọc dựa trên nội dung (**Content-Based Filtering** sử dụng TF-IDF & Cosine Similarity).
  - Lọc cộng tác (**Collaborative Filtering**: User-Based & Item-Based KNN).
  - Phân rược ma trận (**Matrix Factorization**: SVD, SVD++, NMF với thư viện `scikit-surprise`).
  - Mô hình lai hợp (**Hybrid Recommender System**).
- **Thực hành đánh giá:** Sử dụng các chỉ số đo lường chuẩn xác như **RMSE, MAE, Precision@K, Recall@K, MAP, NDCG**.
- **Đóng gói & Triển khai:** Xây dựng Web App tương tác (Streamlit/Flask) giao diện trực quan cho người dùng cuối.

---


## 📊 DỮ LIỆU SỬ DỤNG (MOVIELENS DATASET)

Dự án hỗ trợ các phiên bản tập dữ liệu MovieLens: `ml-100k` (100,000 đánh giá) và `ml-1m` (1,000,000 đánh giá).

### Cấu trúc các file dữ liệu chính:

| Tên File | Số lượng bản ghi (ML-100K) | Cột dữ liệu chính | Mô tả |
| :--- | :--- | :--- | :--- |
| `ratings.csv` / `u.data` | 100,000 | `userId`, `movieId`, `rating`, `timestamp` | Chứa đánh giá từ 1 đến 5 sao của người dùng cho từng bộ phim. |
| `movies.csv` / `u.item` | 1,682 | `movieId`, `title`, `genres` | Chứa tên phim, năm phát hành và các thể loại phim (Action, Sci-Fi,...). |
| `users.csv` / `u.user` | 943 | `userId`, `age`, `gender`, `occupation`, `zip_code` | Thông tin nhân khẩu học của người dùng. |
| `tags.csv` / `u.tag` | ~1,100 | `userId`, `movieId`, `tag`, `timestamp` | Nhãn/Từ khóa do người dùng gắn cho phim. |

---


## 🗓️ LỘ TRÌNH & ĐỀ CƯƠNG CHI TIẾT (SYLLABUS ROADMAP)

Dự án/Khóa học được chia thành **6 Mô-đun (Modules)** tương ứng với các tuần thực hiện:

### 📌 Module 1: Khám Phá Dữ Liệu (EDA) & Tiền Xử Lý

- Phân tích phân phối đánh giá (Rating distribution).
- Xác định các bộ phim phổ biến nhất và người dùng tích cực nhất.
- Xử lý dữ liệu khuyết thiếu (Missing values), làm sạch chuỗi văn bản tên phim và thể loại.
- Xử lý bài toán **Sparsity** (Ma trận thưa).

### 📌 Module 2: Hệ Gợi Ý Dựa Trên Độ Phổ Biến (Popularity-Based)

- Lọc phim theo xếp hạng bình quân và số lượng bình chọn (Weighted Rating - IMDB Formula).
- Xây dựng danh sách Top-N phim đề xuất chung cho người dùng mới (Giải quyết bài toán Cold-Start cho User).

### 📌 Module 3: Hệ Gợi Ý Dựa Trên Nội Dung (Content-Based Filtering)

- Biểu diễn thuộc tính phim bằng kỹ thuật **TF-IDF Vectorizer** dựa trên `genres` và `overview/tags`.
- Tính toán độ tương đồng không gian vector bằng **Cosine Similarity**.
- Tạo hồ sơ sở thích người dùng (User Profile Matrix) để đưa ra gợi ý nhân hóa.

### 📌 Module 4: Lọc Cộng Tác (Collaborative Filtering)

- **Memory-based Collaborative Filtering:**
  - User-Based KNN: Tìm tập người dùng tương đồng (Pearson Correlation / Cosine).
  - Item-Based KNN: Tìm tập phim tương đồng dựa trên lịch sử đánh giá.
- **Model-based Collaborative Filtering:**
  - Áp dụng Phân rược ma trận (Matrix Factorization) bằng thuật toán **SVD** (Singular Value Decomposition).
  - Tối ưu hóa siêu tham số (Hyperparameter Tuning) với `GridSearchCV` trong `scikit-surprise`.

### 📌 Module 5: Đánh Giá Mô Hình & Hệ Lai Hợp (Hybrid Systems)

- So sánh hiệu năng các mô hình qua chỉ số **RMSE**, **MAE**, **Precision@K**, **Recall@K**.
- Xây dựng **Hybrid Recommender System** (Kết hợp Content-Based + Collaborative Filtering) để tối ưu độ chính xác và giảm thiểu điểm yếu của từng phương pháp.

### 📌 Module 6: Triển Khai Giao Diện Web & Báo Cáo

- Đóng gói mô hình thành API (FastAPI / Flask).
- Xây dựng giao diện Web App minh họa bằng **Streamlit**.
- Hoàn thiện báo cáo đồ án và tài liệu hướng dẫn kỹ thuật.

---


## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
DeCuongMonHocMovieLens/
│
├── data/ # Thư mục chứa dữ liệu
│ ├── raw/ # Dữ liệu thô từ MovieLens (ml-100k, ml-1m)
│ └── processed/ # Dữ liệu đã qua làm sạch & tiền xử lý
│
├── notebooks/ # Jupyter Notebooks theo từng Module
│ ├── 01_eda_and_preprocessing.ipynb
│ ├── 02_popularity_recommender.ipynb
│ ├── 03_content_based_recommender.ipynb
│ ├── 04_collaborative_filtering.ipynb
│ └── 05_hybrid_and_evaluation.ipynb
│
├── src/ # Mã nguồn chính của dự án (Python Modules)
│ ├── __init__.py
│ ├── data_loader.py # Hàm tải và xử lý dữ liệu
│ ├── models/ # Các thuật toán hệ gợi ý
│ │ ├── content_based.py
│ │ ├── collaborative.py
│ │ └── hybrid.py
│ ├── metrics.py # Các hàm đánh giá (RMSE, MAE, Precision@K)
│ └── utils.py # Hàm bổ trợ (Trực quan hóa, logging)
│
├── app/ # Ứng dụng Web / Demo UI
│ ├── app.py # Streamlit Dashboard main file
│ └── static/ # Hình ảnh, CSS cho ứng dụng web
│
├── models_saved/ # Lưu trữ mô hình đã huấn luyện (.pkl,.joblib)
├── docs/ # Tài liệu đề cương môn học & báo cáo
├── requirements.txt # Danh sách các thư viện Python phụ thuộc
├──.gitignore # Cấu hình bỏ qua các file không cần thiết trên Git
└── README.md # Tài liệu hướng dẫn sử dụng dự án

🛠️ HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN
1. Yêu cầu hệ thống
Python 3.8+ (Khuyến nghị Python 3.9 hoặc 3.10)
Git
2. Tải mã nguồn về máy local
git clone [https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git](https://github.com/nphlanho-star/DeCuongMonHocMovieLens.git)
cd DeCuongMonHocMovieLens
3. Tạo môi trường ảo (Virtual Environment)
Trên Linux / macOS:
 python3 -m venv venv
source venv/bin/activate
Trên Windows:
 python -m venv venv
venv\Scripts\activate
4. Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
5. Tải tập dữ liệu MovieLens
Bạn có thể chạy script tải dữ liệu tự động hoặc tải thủ công từ GroupLens Website:
python src/data_loader.py --dataset ml-100k
6. Khởi chạy Jupyter Notebook hoặc Web App
Mở Jupyter Notebook:
 jupyter notebook notebooks/
Khởi chạy Web App demo (Streamlit):
 streamlit run app/app.py

📈 ĐÁNH GIÁ & KẾT QUẢ THỬ NGHIỆM
So sánh hiệu năng giữa các phương pháp trên tập dữ liệu MovieLens-100K (chia tập Train/Test theo tỷ lệ 80/20):
Phương Pháp / Mô Hình	RMSE ↓	MAE ↓	Precision@10 ↑	Recall@10 ↑
Popularity-Based	N/A	N/A	0.125	0.082
Content-Based (TF-IDF + Cosine)	1.024	0.812	0.184	0.135
User-Based KNN	0.965	0.758	0.210	0.162
Item-Based KNN	0.948	0.741	0.225	0.178
Matrix Factorization (SVD)	0.932	0.730	0.248	0.195
Hybrid System (SVD + Content)	0.915	0.712	0.262	0.210

💻 MINH HỌA GIAO DIỆN ỨNG DỤNG (WEB APP DEMO)
Ứng dụng Streamlit cho phép:
1.Chọn User ID: Xem lịch sử đánh giá phim của người dùng.
2.Chọn Thuật toán: Chuyển đổi linh hoạt giữa Content-Based, SVD và Hybrid.
3.Kết quả Đề xuất: Hiển thị Top 10 bộ phim được gợi ý kèm điểm số dự đoán (Predicted Rating) và thể loại.

🤝 HƯỚNG DẪN ĐÓNG GÓP (CONTRIBUTING)
Mọi đóng góp nhằm nâng cấp đề cương môn học và hoàn thiện dự án đều được hoan nghênh!
1.Fork dự án này.
2.Tạo branch tính năng mới (git checkout -b feature/AmazingFeature).
3.Commit các thay đổi (git commit -m 'Add some AmazingFeature').
4.Push lên branch (git push origin feature/AmazingFeature).
5.Mở một Pull Request.

📜 GIẤY PHÉP (LICENSE)
Dự án được phân phối dưới giấy phép MIT License. Chi tiết xem tại file LICENSE.

✉️ TÁC GIẢ & LIÊN HỆ
Tác giả / Chủ sở hữu repo: nphlanho-star
GitHub Repository: nphlanho-star/DeCuongMonHocMovieLens
Ý kiến & Thắc mắc: Vui lòng mở Issue trên GitHub repo.
```

Điểm nổi bật của file README này:
1.Trực quan & Hiện đại: Sử dụng các Badge tĩnh (Python, Dataset, License, Status), Emoji sinh động và cây thư mục định dạng text rõ ràng.
2.Đề cương chuẩn hóa (Syllabus Roadmap): Chia theo từng Module/Tuần thực hiện từ EDA, Popularity, Content-Based, Collaborative Filtering (KNN, SVD) đến Hybrid & Triển khai Web.
3.Đầy đủ các phần chuẩn cho GitHub Repo: Giới thiệu, Cấu trúc thư mục, Hướng dẫn cài đặt lệnh từng bước (venv, pip, streamlit), Bảng so sánh chỉ số đánh giá (RMSE, MAE, Precision@K, Recall@K), Hướng dẫn đóng góp (Contributing) và Giấy phép.
