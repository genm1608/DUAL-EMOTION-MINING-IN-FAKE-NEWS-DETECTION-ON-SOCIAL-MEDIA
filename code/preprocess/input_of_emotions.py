import os
import json
from tqdm import tqdm
import time
import numpy as np
import sys
sys.path.append('../emotion')
import extract_emotion_ch_configurable as extract_emotion_ch
import extract_emotion_en_configurable as extract_emotion_en

# --- BỔ SUNG: Lấy version từ môi trường ---
EMOTION_VER = os.environ.get('EMOTION_VERSION', 'v1') 

save_dir = './data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datasets_ch = ['Weibo-16', 'Weibo-20']
datasets_en = ['RumourEval-19']

for dataset in datasets_ch + datasets_en:
    print('\n\n{} [{}]\tProcessing the dataset: {} (Version: {})\n'.format(
        '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dataset, EMOTION_VER))

    if dataset in datasets_ch:
        extract_pkg = extract_emotion_ch
    else:
        extract_pkg = extract_emotion_en

    data_dir = os.path.join('../../dataset', dataset)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    emotion_dir = os.path.join(output_dir, 'emotions')
    if not os.path.exists(emotion_dir):
        os.mkdir(emotion_dir)

    # --- SỬA ĐỔI: Tạo thư mục con cho từng Version (v1, v2, v8...) ---
    version_dir = os.path.join(emotion_dir, EMOTION_VER)
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)

    split_datasets = [json.load(open(os.path.join(
        data_dir, '{}.json'.format(t)), 'r')) for t in ['train', 'val', 'test']]
    split_datasets = dict(zip(['train', 'val', 'test'], split_datasets))

    for t, pieces in split_datasets.items():
        # SỬA ĐỔI logic kiểm tra: Kiểm tra file trong version_dir thay vì output_dir
        arr_is_saved = any('.npy' in f and t in f for f in os.listdir(version_dir)) if os.path.exists(version_dir) else False
        json_is_saved = any(t in f and f.endswith('.json') for f in os.listdir(output_dir))

        if arr_is_saved:
            print(f'  [SKIP] {t} version {EMOTION_VER} already exists.')
            continue

        if json_is_saved:
            pieces = json.load(open(os.path.join(output_dir, '{}.json'.format(t)), 'r'))

        # words cutting
        if 'content_words' not in pieces[0].keys():
            print('[{}]\tWords Cutting...'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            for p in tqdm(pieces):
                p['content_words'] = extract_pkg.cut_words_from_text(p['content'])
                p['comments_words'] = [extract_pkg.cut_words_from_text(com) for com in p['comments']]
            with open(os.path.join(output_dir, '{}.json'.format(t)), 'w') as f:
                json.dump(pieces, f, indent=4, ensure_ascii=False)

        # Trích xuất đặc trưng cảm xúc
        emotion_arr = [extract_pkg.extract_dual_emotion(p) for p in tqdm(pieces)]
        emotion_arr = np.array(emotion_arr)
        
        # SỬA ĐỔI: Lưu vào version_dir
        save_path = os.path.join(version_dir, '{}_{}.npy'.format(t, emotion_arr.shape))
        np.save(save_path, emotion_arr)
        
        print('{} dataset: got a {} emotion arr -> Saved to {}'.format(t, emotion_arr.shape, save_path))

        extract_emotion_ch.print_global_summary()
        extract_emotion_ch.reset_global_stats()

        print(f'\n[Done] {t} dataset: got a {emotion_arr.shape} emotion arr')