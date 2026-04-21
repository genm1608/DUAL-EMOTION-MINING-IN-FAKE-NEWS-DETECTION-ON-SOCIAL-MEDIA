"""
analyze_dutir_weibo.py
-----------------------
Phân tích tần suất xuất hiện của 21 nhãn DUTIR trong dataset Weibo-16.

Mục tiêu:
  1. Đếm số lần mỗi nhãn xuất hiện trong TEXT (bài đăng) và COMMENTS (bình luận)
  2. So sánh phân phối nhãn giữa tin GIẢ và tin THẬT
  3. Tính discriminative score → căn cứ để chọn nhãn nào thêm vào

Kết quả:
  - In bảng tần suất ra màn hình
  - Lưu CSV để dùng trong luận văn

Cách chạy (từ thư mục code/preprocess):
  python analyze_dutir_weibo.py

Yêu cầu:
  - Đã có dataset/Weibo-16/train.json, val.json, test.json
  - Đã có resources/Chinese/大连.../preprocess/words2array_27351.pkl
"""

import os
import sys
import json
import joblib
import jieba
import numpy as np
import pandas as pd
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR  = '../../dataset/Weibo-16'
DALIAN_PKL   = '../../resources/Chinese/大连理工大学情感词汇本体库/preprocess/words2array_27351.pkl'
OUTPUT_CSV   = './dutir_analysis_weibo16.csv'

# 21 nhãn DUTIR và tên tiếng Việt
DUTIR_LABELS = {
    'PH': 'Vui vẻ/Hạnh phúc',
    'PA': 'Yêu thích',
    'PB': 'Hy vọng',
    'PC': 'Tự hào',
    'PD': 'Bình an/Tin tưởng',
    'PE': 'Ngưỡng mộ',
    'PF': 'Biết ơn',
    'PG': 'Dũng cảm',
    'PK': 'Ngạc nhiên (tích cực)',
    'NN': 'Tức giận',
    'NA': 'Buồn bã',
    'NB': 'Sợ hãi',
    'NC': 'Ghê tởm',
    'ND': 'Xấu hổ',
    'NE': 'Thất vọng',
    'NF': 'Tội lỗi',
    'NG': 'Ghen tuông',
    'NH': 'Chán ghét',
    'NI': 'Bi quan',
    'NJ': 'Lo lắng',
    'NL': 'Ngạc nhiên (tiêu cực)',
    'NK': 'Kinh ngạc',
}

# ─── Load từ điển DUTIR ───────────────────────────────────────────────────────

def load_dutir(pkl_path):
    label_dict, words_dict = joblib.load(pkl_path)
    # label_dict: {'NA': 0, 'NB': 1, ...} — index của từng nhãn
    # words_dict: {'词': array(29,)} — vector 29 chiều
    # Chiều 0-20: one-hot 21 nhãn; chiều 21-28: intensity/score/aux
    label_index = {label: idx for idx, label in enumerate(sorted(label_dict.keys()))}
    return label_index, words_dict

# ─── Phân tích 1 đoạn text ────────────────────────────────────────────────────

def count_labels_in_text(text, words_dict, label_index):
    """
    Đếm số lần mỗi nhãn xuất hiện trong text.
    Trả về dict: {nhãn: count}
    """
    if not text or not isinstance(text, str):
        return defaultdict(int)

    counts = defaultdict(int)
    words  = jieba.lcut(text)

    for word in words:
        if word in words_dict:
            vec = words_dict[word]
            # Chiều 0-20 là 21 nhãn (sorted: NA, NB, NC, ND, NE, NG, NH, NI, NJ, NK, NL, NN, PA, PB, PC, PD, PE, PF, PG, PH, PK)
            for label, idx in label_index.items():
                if idx < len(vec) and vec[idx] > 0:
                    counts[label] += 1
    return counts


def count_labels_in_comments(comments, words_dict, label_index):
    """Gộp counts từ tất cả bình luận."""
    total = defaultdict(int)
    for comment in comments:
        c = count_labels_in_text(comment, words_dict, label_index)
        for label, cnt in c.items():
            total[label] += cnt
    return total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('PHÂN TÍCH TẦN SUẤT 21 NHÃN DUTIR TRONG WEIBO-16')
    print('=' * 70)

    # Load từ điển
    print('\n[1] Load từ điển DUTIR...')
    label_index, words_dict = load_dutir(DALIAN_PKL)
    labels_sorted = sorted(label_index.keys())
    print(f'    Số từ: {len(words_dict):,} | Số nhãn: {len(label_index)}')
    print(f'    Thứ tự nhãn: {labels_sorted}')

    # Load data
    print('\n[2] Load ...')
    all_pieces = []
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(DATASET_DIR, f'{split}.json')
        if os.path.exists(fpath):
            pieces = json.load(open(fpath, 'r', encoding='utf-8'))
            all_pieces.extend(pieces)
            print(f'    {split}: {len(pieces)} mẩu tin')

    print(f'    Tổng: {len(all_pieces)} mẩu tin')

    # ── Đếm tần suất ──
    print('\n[3] Đang đếm tần suất nhãn (có thể mất vài phút)...')

    # Accumulator: tách theo fake/real, tách theo text/comment
    stats = {
        'fake': {'text': defaultdict(int), 'comment': defaultdict(int)},
        'real': {'text': defaultdict(int), 'comment': defaultdict(int)},
    }
    count_total  = {'fake': 0, 'real': 0}
    count_errors = 0

    for i, piece in enumerate(all_pieces):
        if i % 500 == 0:
            print(f'    {i}/{len(all_pieces)}...')

        # Nhãn
        label = piece.get('label', '')
        if label not in ('fake', 'real'):
            # Weibo-16 có thể dùng 0/1
            label_raw = piece.get('label', -1)
            if label_raw in (0, '0'):
                label = 'fake'
            elif label_raw in (1, '1'):
                label = 'real'
            else:
                count_errors += 1
                continue

        count_total[label] += 1

        # Text content
        content = piece.get('content', '') or piece.get('text', '')
        text_counts = count_labels_in_text(content, words_dict, label_index)
        for lbl, cnt in text_counts.items():
            stats[label]['text'][lbl] += cnt

        # Comments
        comments = piece.get('comments', [])
        if comments:
            # Comments có thể là list of str hoặc list of dict
            if isinstance(comments[0], dict):
                comments = [c.get('content', '') or c.get('text', '')
                            for c in comments]
            comment_counts = count_labels_in_comments(
                comments, words_dict, label_index)
            for lbl, cnt in comment_counts.items():
                stats[label]['comment'][lbl] += cnt

    if count_errors > 0:
        print(f'    [WARNING] {count_errors} mẩu tin bỏ qua (không có nhãn hợp lệ)')

    print(f'\n    Fake: {count_total["fake"]} | Real: {count_total["real"]}')

    # ── Tính kết quả ──
    print('\n[4] Tính toán kết quả...')

    rows = []
    for label_code in labels_sorted:
        label_name = DUTIR_LABELS.get(label_code, label_code)

        fake_text    = stats['fake']['text'][label_code]
        real_text    = stats['real']['text'][label_code]
        fake_comment = stats['fake']['comment'][label_code]
        real_comment = stats['real']['comment'][label_code]

        total_text    = fake_text + real_text
        total_comment = fake_comment + real_comment
        total_all     = total_text + total_comment

        # Tần suất chuẩn hóa (trên mỗi tin)
        n_fake = max(count_total['fake'], 1)
        n_real = max(count_total['real'], 1)

        fake_text_rate    = fake_text    / n_fake
        real_text_rate    = real_text    / n_real
        fake_comment_rate = fake_comment / n_fake
        real_comment_rate = real_comment / n_real

        # Discriminative score = |fake_rate - real_rate| / (fake_rate + real_rate + 1e-9)
        # Càng cao → nhãn này càng phân biệt được fake/real
        disc_text    = abs(fake_text_rate - real_text_rate) / \
                       (fake_text_rate + real_text_rate + 1e-9)
        disc_comment = abs(fake_comment_rate - real_comment_rate) / \
                       (fake_comment_rate + real_comment_rate + 1e-9)
        disc_avg     = (disc_text + disc_comment) / 2

        rows.append({
            'Nhãn':             label_code,
            'Tên':              label_name,
            'Fake_Text':        fake_text,
            'Real_Text':        real_text,
            'Fake_Comment':     fake_comment,
            'Real_Comment':     real_comment,
            'Tổng_Text':        total_text,
            'Tổng_Comment':     total_comment,
            'Tổng_Tất_Cả':      total_all,
            'Fake_Text_Rate':   round(fake_text_rate,    3),
            'Real_Text_Rate':   round(real_text_rate,    3),
            'Fake_Cmt_Rate':    round(fake_comment_rate, 3),
            'Real_Cmt_Rate':    round(real_comment_rate, 3),
            'Disc_Text':        round(disc_text,    3),
            'Disc_Comment':     round(disc_comment, 3),
            'Disc_Avg':         round(disc_avg,     3),
        })

    df = pd.DataFrame(rows)

    # ── In bảng tần suất ──
    print('\n' + '=' * 70)
    print('BẢNG 1: TẦN SUẤT XUẤT HIỆN (sắp xếp theo tổng tất cả)')
    print('=' * 70)
    df_freq = df.sort_values('Tổng_Tất_Cả', ascending=False)
    print(f'{"Nhãn":<6} {"Tên":<28} {"Tổng Text":>10} {"Tổng Cmt":>10} {"Tổng":>10}')
    print('-' * 70)
    for _, row in df_freq.iterrows():
        print(f'{row["Nhãn"]:<6} {row["Tên"]:<28} '
              f'{row["Tổng_Text"]:>10,} {row["Tổng_Comment"]:>10,} '
              f'{row["Tổng_Tất_Cả"]:>10,}')

    print('\n' + '=' * 70)
    print('BẢNG 2: KHẢ NĂNG PHÂN BIỆT FAKE/REAL (sắp xếp theo Disc_Avg)')
    print('Disc_Avg cao → nhãn này phân biệt fake/real tốt → NÊN CHỌN')
    print('=' * 70)
    df_disc = df.sort_values('Disc_Avg', ascending=False)
    print(f'{"Nhãn":<6} {"Tên":<28} {"Fake/Tin":>9} {"Real/Tin":>9} '
          f'{"Disc_Text":>10} {"Disc_Cmt":>10} {"Disc_Avg":>10}')
    print('-' * 70)
    for _, row in df_disc.iterrows():
        # Dùng text rate để hiển thị
        print(f'{row["Nhãn"]:<6} {row["Tên"]:<28} '
              f'{row["Fake_Text_Rate"]:>9.3f} {row["Real_Text_Rate"]:>9.3f} '
              f'{row["Disc_Text"]:>10.3f} {row["Disc_Comment"]:>10.3f} '
              f'{row["Disc_Avg"]:>10.3f}')

    print('\n' + '=' * 70)
    print('KHUYẾN NGHỊ CHỌN NHÃN (dựa trên Disc_Avg + Tổng_Tất_Cả)')
    print('=' * 70)
    # Top nhãn theo disc_avg và tần suất đủ cao (tổng > median)
    median_freq = df['Tổng_Tất_Cả'].median()
    df_recommend = df[df['Tổng_Tất_Cả'] > median_freq].sort_values(
        'Disc_Avg', ascending=False)
    print('Nhãn đủ phổ biến VÀ có khả năng phân biệt cao nhất:')
    for rank, (_, row) in enumerate(df_recommend.head(10).iterrows(), 1):
        overlap = '← Baidu có tương đương' if row['Nhãn'] in \
            ('NN', 'NC', 'NB', 'PH', 'NA', 'PK', 'PA', 'NI') else '← MỚI so với Baidu'
        print(f'  {rank:2d}. {row["Nhãn"]} ({row["Tên"]:<25}) '
              f'Disc={row["Disc_Avg"]:.3f}  {overlap}')

    # ── Lưu CSV ──
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f'\n[Saved] → {OUTPUT_CSV}')
    print('\nDùng file CSV này để vẽ biểu đồ hoặc trình bày trong luận văn.')


if __name__ == '__main__':
    main()