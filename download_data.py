import pandas as pd
import requests
import re
import os
import time

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("📂 Đang kiểm tra dữ liệu hiện có...")

# Read file gốc
if not os.path.exists('movies_mapped.csv'):
    print("❌ Lỗi: Không tìm thấy file movies_mapped.csv!")
    exit()

base_df = pd.read_csv('movies_mapped.csv')

# Kiểm tra nếu đã có file tải dở dang thì đọc tiếp, chưa có thì tạo mới
if os.path.exists('movies_enriched.csv'):
    enriched_df = pd.read_csv('movies_enriched.csv')
    print("🔄 Tìm thấy file 'movies_enriched.csv' cũ. Sẽ tiếp tục tải các phim còn thiếu...")
else:
    enriched_df = base_df.copy()
    enriched_df['overview'] = ""
    enriched_df['poster_url'] = ""

if 'overview' not in enriched_df.columns: enriched_df['overview'] = ""
if 'poster_url' not in enriched_df.columns: enriched_df['poster_url'] = ""

total = len(enriched_df)
count_downloaded = 0

print(f"🚀 Bắt đầu tiến trình tải dữ liệu cho {total} bộ phim...")

for idx, row in enriched_df.iterrows():
    # BỎ QUA nếu phim này đã có dữ liệu rồi (Cơ chế Resume)
    current_ov = str(row['overview'])
    current_img = str(row['poster_url'])
    if current_ov != "" and current_ov != "nan" and current_ov != "No overview available." and current_img.startswith(
            "http"):
        continue

    clean_name = re.sub(r'\s*\(\d{4}\)', '', str(row['title'])).strip()
    if ", The" in clean_name:
        clean_name = "The " + clean_name.replace(", The", "").strip()
    elif ", A" in clean_name:
        clean_name = "A " + clean_name.replace(", A", "").strip()
    elif ", An" in clean_name:
        clean_name = "An " + clean_name.replace(", An", "").strip()

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={requests.utils.quote(clean_name)}"

    overview_val = "No overview available."
    poster_val = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"

    try:
        res = requests.get(url, headers=headers, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            if data.get('results') and len(data['results']) > 0:
                ov = data['results'][0].get('overview', '')
                if ov and ov.strip(): overview_val = ov.strip()

                p_path = data['results'][0].get('poster_path')
                if p_path: poster_val = f"https://image.tmdb.org/t/p/w500{p_path}"
    except Exception:
        pass

    enriched_df.at[idx, 'overview'] = overview_val
    enriched_df.at[idx, 'poster_url'] = poster_val
    count_downloaded += 1

    # 💾 CỨ 50 PHIM LÀ TỰ ĐỘNG GHI LƯU VÀO Ổ CỨNG 1 LẦN
    if count_downloaded % 50 == 0 or (idx + 1) == total:
        enriched_df.to_csv('movies_enriched.csv', index=False, encoding='utf-8')
        pct = (idx + 1) / total * 100
        print(f"✅ [Đã lưu Checkpoint] Tiến độ: {idx + 1}/{total} phim ({pct:.1f}%)")

print("\n🎉 XONG! Toàn bộ dữ liệu đã được tải và lưu an toàn vào 'movies_enriched.csv'.")