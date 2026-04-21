"""
extract_emotion_en_configurable.py
------------------------------------
Phiên bản có thể cấu hình của extract_emotion_en.py.
Thay thế NVIDIA category bằng NRC EmoLex với các mức độ khác nhau.

CÁCH DÙNG:
  Đặt biến môi trường EMOTION_VERSION trước khi chạy:
    v1 → NVIDIA 16 chiều (gốc)
    v2 → NRC 8 nhãn thay NVIDIA           (publisher=46)
    v3 → NRC 8 + NRC intensity 8 + sent 2  (publisher=38, tích hợp sâu hơn)

TÍCH HỢP:
  Trong input_of_emotions.py, thay:
    import extract_emotion_en
  Bằng:
    import extract_emotion_en_configurable as extract_emotion_en
"""

import os
import re
import nltk
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─── Chọn phiên bản ───────────────────────────────────────────────────────────

EMOTION_VERSION = os.environ.get('EMOTION_VERSION', 'v1')
assert EMOTION_VERSION in ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'), \
    "EMOTION_VERSION phải là v1 đến v8"
    
    
_VERSION_DESC = {
    'v1': 'NVIDIA 16 chiều (gốc)',
    'v2': 'NRC 8 nhãn thay NVIDIA → publisher=46',
    'v3': 'NRC lex+int tích hợp → publisher=38',
    'v4': 'NRC lex+int → publisher=38',
    'v5': 'NRC lex+int → publisher=38',
    'v6': 'NRC lex+int → publisher=38',
    'v7': 'NRC lex+int → publisher=38',
    'v8': 'NRC lex+int → publisher=38',
}
print(f"\n[EmotionConfig-EN] version='{EMOTION_VERSION}': {_VERSION_DESC[EMOTION_VERSION]}")

# ─── NRC Resources ────────────────────────────────────────────────────────────

_lex_cats, _lex_words = joblib.load(
    '../../resources/English/NRC/preprocess/preprocess-lexicon.pkl')
_int_cats, _int_words = joblib.load(
    '../../resources/English/NRC/preprocess/preprocess-intensity.pkl')

# 8 emotion categories (bỏ positive/negative khỏi lexicon)
NRC_EMOTION_CATS = [c for c in _lex_cats if c not in ('positive', 'negative')]
NRC_EMOTION_CATS.sort()
_nrc_cat2idx = {c: i for i, c in enumerate(NRC_EMOTION_CATS)}

nrc_emotion_words = set(_lex_words.keys()).union(set(_int_words.keys()))

print(f'[EN] NRC Lexicon: {len(_lex_words)} từ, {len(_lex_cats)} categories')
print(f'[EN] NRC Intensity: {len(_int_words)} từ, {len(_int_cats)} categories')
print(f'[EN] NRC 8 emotion labels: {NRC_EMOTION_CATS}')

# ─── NVIDIA (chỉ dùng cho v1) ─────────────────────────────────────────────────

nvidia_emotions = ['anger', 'anticipation', 'disgust',
                   'fear', 'joy', 'sadness', 'surprise', 'trust']
nvidia_emotions.sort()


def nvidia_arr(emotions_labels_dict, emotions_probs_dict):
    """16 chiều NVIDIA (8 label + 8 prob) — dùng cho v1."""
    arr = np.zeros(len(nvidia_emotions) * 2)
    if emotions_labels_dict is None or emotions_probs_dict is None:
        return arr
    for i, e in enumerate(nvidia_emotions):
        arr[i]                    = emotions_labels_dict[e]
        arr[i + len(nvidia_emotions)] = emotions_probs_dict[e]
    return arr

# ─── Degree / Negation words ──────────────────────────────────────────────────

negation_words = []
with open('../../resources/English/others/negative/negationWords.txt', 'r') as src:
    for line in src:
        negation_words.append(line.strip())

how_words_dict = {}
with open('../../resources/English/HowNet/intensifierWords.txt', 'r') as src:
    for line in src:
        parts = line.strip().split()
        how_words_dict[' '.join(parts[:-1])] = float(parts[-1])

print(f'[EN] Negation: {len(negation_words)}, Degree: {len(how_words_dict)}')


def get_not_and_how_value(cut_words, i, windows=4):
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

# ─── NRC functions ────────────────────────────────────────────────────────────

def nrc_category_arr(cut_words, windows=4):
    """
    8 chiều NRC emotion category (thay thế NVIDIA cho v2/v3).
    Đếm số từ mang từng emotion, có xét negation và degree words.
    """
    arr = np.zeros(8)
    for i, word in enumerate(cut_words):
        if word in _lex_words:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            # Chỉ lấy 8 emotion (bỏ positive/negative ở index 5,6)
            lex_vec = _lex_words[word]  # shape (10,)
            for j, cat in enumerate(NRC_EMOTION_CATS):
                orig_idx = _lex_cats.index(cat)
                arr[j] += not_v * how_v * lex_vec[orig_idx]
    return arr


def nrc_full_arr(cut_words, windows=4):
    """
    18 chiều NRC đầy đủ: lexicon(10) + intensity(8) — giống code gốc.
    Dùng cho v1 và v2 (phần lexicon/intensity không đổi).
    """
    arr = np.zeros(len(_lex_cats) + len(_int_cats))
    for i, word in enumerate(cut_words):
        if word in nrc_emotion_words:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            if word in _lex_words:
                arr[:len(_lex_cats)] += not_v * how_v * _lex_words[word]
            if word in _int_words:
                arr[len(_lex_cats):] += not_v * how_v * _int_words[word]
    return arr

# ─── Sentiment Score ──────────────────────────────────────────────────────────

sentiment_analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    scores = sentiment_analyzer.polarity_scores(text)
    return scores['pos'], scores['neg'], scores['neu'], scores['compound']

# ─── Auxiliary Features ───────────────────────────────────────────────────────

def isEmoji(c):
    return (u'\U0001F600' <= c <= u'\U0001F64F' or
            u'\U0001F300' <= c <= u'\U0001F5FF' or
            u'\U0001F680' <= c <= u'\U0001F6FF' or
            u'\U0001F1E0' <= c <= u'\U0001F1FF')


def emoji_count(text):
    return sum(1 for c in text if isEmoji(c)) / len(text)


smiling_emoticons  = [':-)', ':)', ':o)', ':]', ':3', ':c)', ':>', '=]',
                      '8)', '=)', ':}', ':^)', ':っ)']
frowning_emoticons = ['>:[', ':-(' , ':(', ':-c', ':c', ':-<', ':っC',
                      ':<', ':-[', ':[', ':{']


def emoticon_arr(text):
    s = sum(text.count(e) for e in smiling_emoticons)  / len(text)
    f = sum(text.count(e) for e in frowning_emoticons) / len(text)
    return s, f, emoji_count(text)


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

pos_words = set(_init_words('../../resources/English/HowNet/正面情感词语（英文）.txt') +
                _init_words('../../resources/English/HowNet/正面评价词语（英文）.txt'))
neg_words = set(_init_words('../../resources/English/HowNet/负面情感词语（英文）.txt') +
                _init_words('../../resources/English/HowNet/负面评价词语（英文）.txt'))
first_pronoun  = _init_words('../../resources/English/others/pronoun/1-personal-pronoun.txt')
second_pronoun = _init_words('../../resources/English/others/pronoun/2-personal-pronoun.txt')
third_pronoun  = _init_words('../../resources/English/others/pronoun/3-personal-pronoun.txt')


def sentiment_words_count(cut_words):
    if not cut_words:
        return [0, 0, 0, 0]
    pos = sum(1 for w in pos_words if w in cut_words) / len(cut_words)
    neg_w = sum(1 for w in neg_words if w in cut_words) / len(cut_words)
    deg = sum(how_words_dict[w] for w in how_words_dict if w in cut_words)
    neg_cnt = sum(cut_words.count(w) for w in negation_words) / len(cut_words)
    return [pos, neg_w, deg, neg_cnt]


def pronoun_count(cut_words):
    if not cut_words:
        return [0, 0, 0]
    return [sum(cut_words.count(w) for w in pron) / len(cut_words)
            for pron in [first_pronoun, second_pronoun, third_pronoun]]


def upper_letter_count(text):
    return sum(1 for c in text if c.isupper()) / len(text)


def auxilary_features(text, cut_words):
    arr = np.zeros(16)
    arr[:3]   = emoticon_arr(text)
    arr[3:8]  = symbols_count(text)
    arr[8:12] = sentiment_words_count(cut_words)
    arr[12:15] = pronoun_count(cut_words)
    arr[15]   = upper_letter_count(text)
    return arr

# ─── Main ─────────────────────────────────────────────────────────────────────

def del_url_at(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    return text.replace('\r', '').replace('\t', '')


def cut_words_from_text(text):
    pattern = r"""(?x)
                  (?:[A-Z]\.)+
                  |\d+(?:\.\d+)?%?
                  |\w+(?:[-']\w+)*
                  |\.\.\.
                  |(?:[.,;"'?():\-_`!])
                """
    return nltk.regexp_tokenize(del_url_at(text), pattern)


def extract_publisher_emotion(content, content_words,
                               emotions_labels_dict, emotions_probs_dict):
    """
    v1 (gốc):  [nvidia(16) | nrc_full(18) | vader(4) | aux(16)] = 54
    v2:        [nrc_cat(8)  | nrc_full(18) | vader(4) | aux(16)] = 46
    v3-v8:     [nrc_cat(8)  | nrc_int(8)   | vader(4) | aux(16)] = 36
    """
    text, cut_words = content, content_words

    if EMOTION_VERSION == 'v1':
        arr = np.zeros(54)
        arr[:16]   = nvidia_arr(emotions_labels_dict, emotions_probs_dict)
        arr[16:34] = nrc_full_arr(cut_words)
        arr[34:38] = sentiment_score(text)
        arr[38:54] = auxilary_features(text, cut_words)

    elif EMOTION_VERSION == 'v2':
        arr = np.zeros(46)
        arr[:8]    = nrc_category_arr(cut_words)
        arr[8:26]  = nrc_full_arr(cut_words)
        arr[26:30] = sentiment_score(text)
        arr[30:46] = auxilary_features(text, cut_words)

    else:  # v3-v8
        arr = np.zeros(36)
        arr[:8]    = nrc_category_arr(cut_words)
        arr[8:16]  = _nrc_intensity_pooled(cut_words)
        arr[16:20] = sentiment_score(text)
        arr[20:36] = auxilary_features(text, cut_words)

    return arr


def _nrc_intensity_pooled(cut_words):
    """8 chiều NRC intensity trung bình cho v3."""
    arr = np.zeros(8)
    count = 0
    for word in cut_words:
        if word in _int_words:
            arr += _int_words[word]
            count += 1
    return arr / max(count, 1)


def _publisher_size():
    return {
        'v1': 54, 'v2': 46,
        'v3': 36, 'v4': 36, 'v5': 36,
        'v6': 36, 'v7': 36, 'v8': 36,
    }[EMOTION_VERSION]
    
PUBLISHER_SIZE = _publisher_size()
print(f'[EN] Publisher vector size: {PUBLISHER_SIZE} | Dual emotion: {PUBLISHER_SIZE * 5}')


def extract_social_emotion(comments, comments_words,
                            mean_emotions_labels_dict, max_emotions_labels_dict,
                            mean_emotions_probs_dict,  max_emotions_probs_dict):
    if len(comments) == 0:
        arr = np.zeros(PUBLISHER_SIZE)
        return arr, arr, np.concatenate([arr, arr])

    arr = np.zeros((len(comments), PUBLISHER_SIZE))
    for i in range(len(comments)):
        arr[i] = extract_publisher_emotion(
            comments[i], comments_words[i], None, None)

    mean_arr = np.mean(arr, axis=0)
    max_arr  = np.max(arr, axis=0)

    if EMOTION_VERSION == 'v1':
        mean_arr[:16] = nvidia_arr(mean_emotions_labels_dict, mean_emotions_probs_dict)
        max_arr[:16]  = nvidia_arr(max_emotions_labels_dict,  max_emotions_probs_dict)
    else:
        # v2/v3: category từ NRC đã được tính từ text, không cần ghi đè API
        pass

    return mean_arr, max_arr, np.concatenate([mean_arr, max_arr])


def extract_dual_emotion(piece, COMMENTS=100):
    publisher_emotion = extract_publisher_emotion(
        piece['content'],
        piece['content_words'],
        piece.get('content_emotions_labels'),
        piece.get('content_emotions_probs'))

    mean_arr, max_arr, social_emotion = extract_social_emotion(
        piece['comments'][:COMMENTS],
        piece['comments_words'][:COMMENTS],
        piece.get('comments100_emotions_labels_mean_pooling'),
        piece.get('comments100_emotions_labels_max_pooling'),
        piece.get('comments100_emotions_probs_mean_pooling'),
        piece.get('comments100_emotions_probs_max_pooling'))

    emotion_gap  = np.concatenate([publisher_emotion - mean_arr,
                                   publisher_emotion - max_arr])
    dual_emotion = np.concatenate([publisher_emotion, social_emotion, emotion_gap])
    return dual_emotion