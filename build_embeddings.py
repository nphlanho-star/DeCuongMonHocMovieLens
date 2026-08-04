import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

print("⏳ Đang khởi tạo và mã hóa không gian Vector SBERT...")
csv_path = 'movies_enriched.csv' if os.path.exists('movies_enriched.csv') else 'movies_mapped.csv'
movies_df = pd.read_csv(csv_path)

if 'overview' not in movies_df.columns:
    movies_df['overview'] = "No overview available."

sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
corpus_texts = (movies_df['title'] + ". " + movies_df['overview'].fillna('')).tolist()

# Encode toàn bộ dữ liệu 1 lần duy nhất
overview_embeddings = sbert_model.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)

# Lưu ma trận Vector ra đĩa cứng
np.save('overview_embeddings.npy', overview_embeddings)
print("✅ Đã lưu ma trận Vector thành công vào file 'overview_embeddings.npy'!")