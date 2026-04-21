"""
extract_emotion_ch_configurable.py
------------------------------------
Phiên bản có thể cấu hình của extract_emotion_ch.py.
Tăng dần số cảm xúc bằng cách cộng thêm nhãn từ từ điển Đại Liên vào Baidu 8.


"""

import os
import joblib
import pandas as pd
import numpy as np
import jieba


global_stats = {
    'publisher': [],
    'social_mean': [],
    'social_max': [],
    'gap': [],
    'dual': []
}


def print_global_summary():
    print("\n" + "="*90)
    print("=== TỔNG HỢP CHỈ SỐ SAU CÙNG CỦA TOÀN DATASET ===")
    print("="*90)

    for name, arr_list in global_stats.items():
        if len(arr_list) == 0:
            continue
        arr = np.array(arr_list)
        print(f"\n{name.upper():15} | samples={len(arr)} | shape={arr.shape}")
        print(f"  Mean   : {arr.mean(axis=0).round(4)}")
        print(f"  Std    : {arr.std(axis=0).round(4)}")
        print(f"  Min    : {arr.min(axis=0).round(4)}")
        print(f"  Max    : {arr.max(axis=0).round(4)}")
        print(f"  Sum    : {arr.sum():.2f}")
    print("="*90 + "\n")


def reset_global_stats():
    global_stats['publisher'] = []
    global_stats['social_mean'] = []
    global_stats['social_max'] = []
    global_stats['gap'] = []
    global_stats['dual'] = []
# ─── Chọn phiên bản ───────────────────────────────────────────────────────────

EMOTION_VERSION = os.environ.get('EMOTION_VERSION', 'v1')

# Các nhãn Đại Liên thêm vào theo từng phiên bản (sắp xếp theo tần suất xuất hiện)
_DALIAN_EXTRA = {
    'v1': [],
    'v2': ['PH', 'NN', 'PA'],
    'v3': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG'],
    'v4': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG',
           'NB', 'PC', 'PD'],
    'v5': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG',
           'NB', 'PC', 'PD', 'NL', 'NC', 'NE'],
    'v6': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG',
           'NB', 'PC', 'PD', 'NL', 'NC', 'NE',
           'PK', 'PF', 'PE'],
    'v7': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG',
           'NB', 'PC', 'PD', 'NL', 'NC', 'NE',
           'PK', 'PF', 'PE', 'NI', 'NJ', 'NA'],
    'v8': ['PH', 'NN', 'PA', 'ND', 'PB', 'PG',
           'NB', 'PC', 'PD', 'NL', 'NC', 'NE',
           'PK', 'PF', 'PE', 'NI', 'NJ', 'NA',
           'NG', 'NH', 'NK'],
}

assert EMOTION_VERSION in _DALIAN_EXTRA, \
    f"EMOTION_VERSION phải là một trong: {list(_DALIAN_EXTRA.keys())}"

DALIAN_EXTRA_LABELS = _DALIAN_EXTRA[EMOTION_VERSION]
EMOTION_NUM = 8 + len(DALIAN_EXTRA_LABELS)   # df

print(f"\n[EmotionConfig] version='{EMOTION_VERSION}' | df={EMOTION_NUM}")

print(f"\n[EmotionConfig-CH] version='{EMOTION_VERSION}' | "
      f"df={EMOTION_NUM} (Baidu 8 + Đại Liên {len(DALIAN_EXTRA_LABELS)})")
if DALIAN_EXTRA_LABELS:
    print(f"[EmotionConfig-CH] Nhãn Đại Liên thêm: {DALIAN_EXTRA_LABELS}")

# ─── Baidu category ───────────────────────────────────────────────────────────

baidu_emotions = ['angry', 'disgusting', 'fearful',
                  'happy', 'sad', 'neutral', 'pessimistic', 'optimistic']
baidu_emotions.sort()
baidu_emotions_2_index = {e: i for i, e in enumerate(baidu_emotions)}


def baidu_arr(emotions_dict):
    """8 chiều Baidu cố định."""
    arr = np.zeros(8)
    if emotions_dict is None:
        return arr
    for k, v in emotions_dict.items():
        k = 'happy' if k == 'like' else k
        if k in baidu_emotions_2_index:
            arr[baidu_emotions_2_index[k]] += v
    return arr


# ─── Đại Liên ─────────────────────────────────────────────────────────────────

negation_words = []
with open('../../resources/Chinese/others/negative/negationWords.txt', 'r') as src:
    for line in src:
        negation_words.append(line.strip())
print(f'[CH] Số từ phủ định: {len(negation_words)}')

how_words_dict = {}
with open('../../resources/Chinese/HowNet/intensifierWords.txt', 'r') as src:
    for line in src:
        parts = line.strip().split()
        how_words_dict[' '.join(parts[:-1])] = float(parts[-1])
print(f'[CH] Số từ mức độ: {len(how_words_dict)}')


def get_not_and_how_value(cut_words, i, windows):
    not_cnt = 0
    how_v   = 1
    left    = max(0, i - windows)
    window_text = ' '.join(cut_words[left:i])
    for w in negation_words:
        if w in window_text:
            not_cnt += 1
    for w in how_words_dict:
        if w in window_text:
            how_v *= how_words_dict[w]
    return (-1) ** not_cnt, how_v


_pkl_path = None
import os as _os
for _root, _dirs, _files in _os.walk('../../resources/Chinese'):
    for _f in _files:
        if _f.endswith('.pkl'):
            _pkl_path = _os.path.join(_root, _f)

_dalian_data             = joblib.load(_pkl_path)
_dalian_emotion_dict     = _dalian_data[0]   # {label: index} 21 loại
_dalian_words2array      = _dalian_data[1]   # {word: vec(29)}

# Index của từng nhãn Đại Liên trong vector 29 chiều
_DALIAN_LABEL2DIM = {label: idx for label, idx in _dalian_emotion_dict.items()}
print(f'[CH] Đại Liên: {len(_dalian_words2array)} từ, '
      f'vector 29 chiều (21 emotion + 8 intensity)')


def dalianligong_arr(cut_words, windows=2):
    """Vector 29 chiều Đại Liên đầy đủ (dùng cho lexicon/intensity gốc)."""
    arr = np.zeros(29)
    for i, word in enumerate(cut_words):
        if word in _dalian_words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr += not_v * how_v * _dalian_words2array[word]
    return arr


def dalian_extra_arr(cut_words, windows=2):
    """
    Chỉ lấy các chiều emotion Đại Liên được chọn theo DALIAN_EXTRA_LABELS.
    Trả về vector shape (len(DALIAN_EXTRA_LABELS),).
    """
    if not DALIAN_EXTRA_LABELS:
        return np.zeros(0)

    arr = np.zeros(len(DALIAN_EXTRA_LABELS))
    for i, word in enumerate(cut_words):
        if word in _dalian_words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            word_vec = _dalian_words2array[word]
            for j, label in enumerate(DALIAN_EXTRA_LABELS):
                dim = _DALIAN_LABEL2DIM[label]
                arr[j] += not_v * how_v * word_vec[dim]
    return arr


# ─── BosonNLP sentiment ────────────────────────────────────────────────────────

boson_words_dict = {}
with open('../../resources/Chinese/BosonNLP/BosonNLP_sentiment_score.txt', 'r') as src:
    for line in src:
        parts = line.strip().split()
        if len(parts) == 2:
            boson_words_dict[parts[0]] = float(parts[1])
print(f'[CH] BosonNLP: {len(boson_words_dict)} từ')


def boson_value(cut_words, windows=2):
    value = 0
    for i, word in enumerate(cut_words):
        if word in boson_words_dict:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value += not_v * how_v * boson_words_dict[word]
    return value


# ─── Auxiliary Features ───────────────────────────────────────────────────────

emoticon_df    = pd.read_csv('../../resources/Chinese/others/emoticon/emoticon.csv')
emoticons      = emoticon_df['emoticon'].tolist()
emoticon_types = sorted(set(emoticon_df['label'].tolist()))
emoticon2label = dict(zip(emoticon_df['emoticon'], emoticon_df['label']))
emoticon2index = {e: i for i, e in enumerate(emoticon_types)}


def emoticon_arr(text, cut_words):
    arr = np.zeros(len(emoticon_types))
    if len(cut_words) == 0:
        return arr
    for emoticon in emoticons:
        if emoticon in text:
            arr[emoticon2index[emoticon2label[emoticon]]] += text.count(emoticon)
    return arr / len(cut_words)


def symbols_count(text):
    n = len(text)
    return [(text.count('!') + text.count('！')) / n,
            (text.count('?') + text.count('？')) / n,
            (text.count(',') + text.count('，')) / n,
            (text.count('.') + text.count('。')) / n,
            (text.count('..') + text.count('。。')) / n]


def _init_words(file):
    with open(file, 'r', encoding='utf-8') as f:
        return list(set(l.strip() for l in f if l.strip()))

pos_words = set(_init_words('../../resources/Chinese/HowNet/正面情感词语（中文）.txt') +
                _init_words('../../resources/Chinese/HowNet/正面评价词语（中文）.txt'))
neg_words = set(_init_words('../../resources/Chinese/HowNet/负面情感词语（中文）.txt') +
                _init_words('../../resources/Chinese/HowNet/负面评价词语（中文）.txt'))

first_pronoun  = _init_words('../../resources/Chinese/others/pronoun/1-personal-pronoun.txt')
second_pronoun = _init_words('../../resources/Chinese/others/pronoun/2-personal-pronoun.txt')
third_pronoun  = _init_words('../../resources/Chinese/others/pronoun/3-personal-pronoun.txt')


def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0, 0]
    pos = sum(1 for w in pos_words if w in cut_words) / len(cut_words)
    neg = sum(1 for w in neg_words if w in cut_words) / len(cut_words)
    deg = sum(how_words_dict[w] for w in how_words_dict if w in cut_words)
    neg_cnt = sum(cut_words.count(w) for w in negation_words) / len(cut_words)
    return [pos, neg, deg, neg_cnt]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]
    return [sum(cut_words.count(w) for w in pron) / len(cut_words)
            for pron in [first_pronoun, second_pronoun, third_pronoun]]


def auxilary_features(text, cut_words):
    arr = np.zeros(17)
    arr[:5]    = emoticon_arr(text, cut_words)
    arr[5:10]  = symbols_count(text)
    arr[10:14] = sentiment_words_count(cut_words)
    arr[14:17] = pronoun_count(cut_words)
    return arr


# ─── Main ─────────────────────────────────────────────────────────────────────

def cut_words_from_text(text):
    return list(jieba.cut(text))

def extract_publisher_emotion(content, content_words, emotions_dict):
    text, cut_words = content, content_words
    size = EMOTION_NUM + 47
    arr = np.zeros(size)

    arr[:8] = baidu_arr(emotions_dict)

    n_extra = len(DALIAN_EXTRA_LABELS)
    if n_extra > 0:
        arr[8:8+n_extra] = dalian_extra_arr(cut_words)

    offset = 8 + n_extra
    arr[offset:offset+29] = dalianligong_arr(cut_words)
    arr[offset+29] = boson_value(cut_words)
    arr[offset+30:] = auxilary_features(text, cut_words)

    global_stats['publisher'].append(arr.copy())
    return arr


def extract_social_emotion(comments, comments_words, mean_emotions_dict, max_emotions_dict):
    size = EMOTION_NUM + 47
    if len(comments) == 0:
        arr = np.zeros(size)
        return arr, arr, np.concatenate([arr, arr])

    arr = np.zeros((len(comments), size))
    for i in range(len(comments)):
        arr[i] = extract_publisher_emotion(comments[i], comments_words[i], None)

    mean_arr = np.mean(arr, axis=0)
    max_arr = np.max(arr, axis=0)

    mean_arr[:8] = baidu_arr(mean_emotions_dict)
    max_arr[:8] = baidu_arr(max_emotions_dict)

    n_extra = len(DALIAN_EXTRA_LABELS)
    if n_extra > 0:
        mean_arr[8:8+n_extra] = np.mean(arr[:, 8:8+n_extra], axis=0)
        max_arr[8:8+n_extra] = np.max(arr[:, 8:8+n_extra], axis=0)

    global_stats['social_mean'].append(mean_arr.copy())
    global_stats['social_max'].append(max_arr.copy())

    return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])


def extract_dual_emotion(piece, COMMENTS=100):
    publisher = extract_publisher_emotion(piece['content'], piece['content_words'], piece['content_emotions'])

    mean_arr, max_arr, social = extract_social_emotion(
        piece['comments'][:COMMENTS],
        piece['comments_words'][:COMMENTS],
        piece['comments100_emotions_mean_pooling'],
        piece['comments100_emotions_max_pooling'])

    gap = np.concatenate([publisher - mean_arr, publisher - max_arr])
    dual = np.concatenate([publisher, social, gap])

    global_stats['gap'].append(gap.copy())
    global_stats['dual'].append(dual.copy())

    return dual