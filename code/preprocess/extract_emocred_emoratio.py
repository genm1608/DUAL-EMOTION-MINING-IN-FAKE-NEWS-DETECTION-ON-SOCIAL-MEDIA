"""
extract_emocred_emoratio.py
────────────────────────────────────────────────────────────────────
Tính EmoCred và EmoRatio theo định nghĩa trong paper gốc
(Zhang et al., WWW 2021):

- EmoRatio  (Ajao et al.)    : ratio = neg_words / pos_words, 1 chiều
- EmoCred   (Giachanou et al.): lexicon (21d) + intensity (21d) = 42 chiều

Chỉ dùng CONTENT (publisher emotion) — không dùng comments.

Cách chạy (từ thư mục code/preprocess/):
    set EMOTION_VERSION=v1
    python extract_emocred_emoratio.py

Output: data/Weibo-16/emotions/{EMOTION_VERSION}/
    train_emocred_(N,42).npy
    val_emocred_(N,42).npy
    test_emocred_(N,42).npy
    train_emoratio_(N,1).npy
    val_emoratio_(N,1).npy
    test_emoratio_(N,1).npy
"""

import os, sys, json, joblib
import numpy as np
from tqdm import tqdm
import jieba

# ── PATH ──────────────────────────────────────────────────────────────
# DATASET     = 'Weibo-16'
DATASET     = 'Weibo-20'
# DATASET     = 'Weibo-20-temporal'
EMO_VER     = os.environ.get('EMOTION_VERSION', 'v1')
DATA_DIR    = os.path.join('data', DATASET)
OUT_DIR     = os.path.join(DATA_DIR, 'emotions', EMO_VER)
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD DUTIR lexicon ────────────────────────────────────────────────
_pkl_path = None
for root, dirs, files in os.walk('../../resources/Chinese'):
    for f in files:
        if f.endswith('.pkl'):
            _pkl_path = os.path.join(root, f)

_dalian_data         = joblib.load(_pkl_path)
_dalian_emotion_dict = _dalian_data[0]   # {label: index} 21 loại
_dalian_words2array  = _dalian_data[1]   # {word: vec(29)}

# 21 nhãn DUTIR theo thứ tự index
N_EMO = 21   # 21 chiều emotion (0-20)
N_INT = 8    # 8 chiều intensity (21-28)

# ── LOAD HowNet pos/neg words ─────────────────────────────────────────
def _load_words(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set(l.strip() for l in f if l.strip())

pos_words = _load_words('../../resources/Chinese/HowNet/正面情感词语（中文）.txt') | \
            _load_words('../../resources/Chinese/HowNet/正面评价词语（中文）.txt')
neg_words = _load_words('../../resources/Chinese/HowNet/负面情感词语（中文）.txt') | \
            _load_words('../../resources/Chinese/HowNet/负面评价词语（中文）.txt')

print(f'[CH] pos_words: {len(pos_words)} | neg_words: {len(neg_words)}')
# print(f'[CH] DUTIR: {len(_dalian_words2array)} từ | 29 chiều')

# ── CẮT TỪ ───────────────────────────────────────────────────────────
def cut(text):
    if not text or not isinstance(text, str):
        return []
    return list(jieba.cut(text))

# ── EMORATIO — 1 chiều ───────────────────────────────────────────────
def calc_emoratio(cut_words):
    """
    EmoRatio = count(neg_words) / max(count(pos_words), 1)
    Theo Ajao et al. 2019 — chỉ dùng content.
    """
    pos_cnt = sum(1 for w in cut_words if w in pos_words)
    neg_cnt = sum(1 for w in cut_words if w in neg_words)
    return np.array([neg_cnt / max(pos_cnt, 1)], dtype=np.float32)

# ── EMOCRED — 42 chiều ───────────────────────────────────────────────
# Định nghĩa lại hằng số nếu cần
N_EMO_8 = 8 

def calc_emocred(cut_words):
    """
    EmoCred = lexicon(8d) + intensity(8d) = 16 chiều
    Chỉ lấy 8 cảm xúc cơ bản để đồng bộ với EmoRatio (Cratio).
    """
    # Khởi tạo vector 8 chiều thay vì 21
    lex = np.zeros(N_EMO_8, dtype=np.float32)   
    ity = np.zeros(N_EMO_8, dtype=np.float32)   

    for word in cut_words:
        if word in _dalian_words2array:
            vec = _dalian_words2array[word]   # Vector gốc (29,)
            
            # Lấy 8 chiều cảm xúc đầu tiên (vị trí 0-7)
            emo_8 = vec[:N_EMO_8]
            lex += emo_8
            
            # Lấy 8 chiều cường độ tương ứng (vị trí 21-28)
            # Trong DUTIR, cường độ thường nằm sau phần cảm xúc (index 21 trở đi)
            if len(vec) >= (21 + N_EMO_8):
                intensity_8 = vec[21 : 21 + N_EMO_8]
                ity += emo_8 * intensity_8
            else:
                # Fallback nếu không đủ chiều dữ liệu
                ity += emo_8

    # Chuẩn hóa theo số lượng từ
    n = max(len(cut_words), 1)
    lex /= n
    ity /= n

    # Trả về vector 16 chiều (8+8)
    return np.concatenate([lex, ity])

# ── XỬ LÝ 1 SPLIT ────────────────────────────────────────────────────
def process_split(split):
    path = os.path.join(DATA_DIR, split + '.json')
    if not os.path.exists(path):
        print(f'[SKIP] Không tìm thấy {path}')
        return

    pieces = json.load(open(path, encoding='utf-8'))
    n = len(pieces)
    print(f'\n[{split}] {n} samples...')

    emocred_arr  = np.zeros((n, 16), dtype=np.float32)
    emoratio_arr = np.zeros((n, 1),  dtype=np.float32)

    for i, piece in enumerate(tqdm(pieces)):
        # Dùng content_words nếu có, không thì cắt từ content
        content_words = piece.get('content_words', None)
        if not content_words:
            content_words = cut(piece.get('content', ''))

        emocred_arr[i]  = calc_emocred(content_words)
        emoratio_arr[i] = calc_emoratio(content_words)

    # Lưu file
    def save(arr, name):
        fname = f'{split}_{name}_{arr.shape}.npy'
        np.save(os.path.join(OUT_DIR, fname), arr)
        print(f'  Saved: {fname}')

    save(emocred_arr,  'emocred')
    save(emoratio_arr, 'emoratio')

# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    print(f'Dataset : {DATASET}')
    print(f'Version : {EMO_VER}')
    print(f'Out dir : {OUT_DIR}')

    for split in ['train', 'val', 'test']:
        process_split(split)

    print('\n[Done]')

if __name__ == '__main__':
    main()
