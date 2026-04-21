"""
train_domain.py
---------------
Phiên bản train.py mở rộng, hỗ trợ domain feature.

Thêm 2 model mới:
  - DomainBiGRU     : BiGRU + Domain (không có emotion)
  - DomainEmoBiGRU  : BiGRU + Domain + Emotion
  - DomainCNN       : CNN + Domain
  - DomainEmoCNN    : CNN + Domain + Emotion

Cách chạy:
  Đặt model trong config_domain.py rồi chạy master_domain.py
  
  Biến môi trường:
    EMOTION_VERSION=v1  (như cũ)
    DOMAIN_TOPICS=10    (số topic LDA, mặc định 10)
"""

import os
import sys
import json
import math
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

# Thêm path đến model mới và model cũ
sys.path.append('../model')
sys.path.append('../../code/model')   # model gốc (BiGRU, MLP)

from BiGRU_domain import DomainAttentionBiGRU
from CNN_domain   import DomainAttentionCNN

# Import model gốc (vẫn dùng được)
try:
    from MLP   import MLP5Layers
    from BiGRU import EmotionEnhancedBiGRU
    from CNN   import CNNModel
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print('[WARNING] Không tìm thấy model gốc — chỉ dùng được DomainAttention models')


# ─── Constants ────────────────────────────────────────────────────────────────

labels_names = ['fake', 'real', 'unverified']
datasets_ch  = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
datasets_en  = ['RumourEval-19']

dataset_dir  = '../../code/preprocess/data'
results_dir  = './results'
os.makedirs(results_dir, exist_ok=True)

EMOTION_VERSION = os.environ.get('EMOTION_VERSION', 'v1')
DOMAIN_TOPICS   = int(os.environ.get('DOMAIN_TOPICS', '10'))

# Tên các model
model_names = [
    'DomainBiGRU',      # 0: BiGRU + Domain (không emotion)
    'DomainEmoBiGRU',   # 1: BiGRU + Domain + Emotion
    'DomainCNN',        # 2: CNN + Domain
    'DomainEmoCNN',     # 3: CNN + Domain + Emotion
]


# ─── Callback ─────────────────────────────────────────────────────────────────

class EpochHistoryLogger(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history.append({
            'epoch':      epoch + 1,
            'train_loss': float(logs.get('loss',         0)),
            'train_acc':  float(logs.get('accuracy',     0)),
            'val_loss':   float(logs.get('val_loss',     0)),
            'val_acc':    float(logs.get('val_accuracy', 0)),
        })


# ─── Metrics ──────────────────────────────────────────────────────────────────

def calculate_RMSE_of_RumourEval(y_pred, test_label):
    errors = []
    for i in range(len(y_pred)):
        pred_label = y_pred[i].argmax()
        if pred_label == 2:
            yhat, confidence = 'unverified', 0.0
        else:
            yhat       = 'fake' if pred_label == 0 else 'real'
            confidence = y_pred[i][pred_label] / np.sum(y_pred[i][:2])
        ground_truth = test_label[i].argmax()
        if pred_label == ground_truth and yhat in ('fake', 'real'):
            errors.append((1 - confidence) ** 2)
        elif ground_truth == 2:
            errors.append(confidence ** 2)
        else:
            errors.append(1.0)
    return math.sqrt(sum(errors) / len(errors))


def predict_single_output(y_pred, test_label):
    y_pred_label = np.zeros(y_pred.shape)
    for i, arg in enumerate(y_pred.argmax(axis=1)):
        y_pred_label[i][arg] = 1
    names    = labels_names[:test_label.shape[-1]]
    accuracy = accuracy_score(test_label, y_pred_label)
    print(f'\nAccuracy: {accuracy:.3f}\n')
    print(classification_report(test_label, y_pred_label,
                                labels=list(range(len(names))),
                                target_names=names, digits=3))
    report = classification_report(test_label, y_pred_label,
                                   labels=list(range(len(names))),
                                   target_names=names, digits=3, output_dict=True)
    return accuracy, report


# ─── Load dataset ─────────────────────────────────────────────────────────────

def load_npy_split(data_dir, split):
    """Load file .npy cho 1 split (train/val/test), bỏ qua embedding_matrix."""
    result = None
    for f in os.listdir(data_dir):
        if not f.endswith('.npy'):
            continue
        if 'embedding_matrix' in f:
            continue
        if f'{split}_' in f:
            result = np.load(os.path.join(data_dir, f))
            break
    return result


def load_domain_split(dataset, split):
    """Load domain vector cho 1 split."""
    domain_dir = os.path.join('../../dataset', dataset, 'domain')
    for f in os.listdir(domain_dir):
        if f.endswith('.npy') and f'{split}_' in f:
            return np.load(os.path.join(domain_dir, f))
    raise FileNotFoundError(
        f'Không tìm thấy domain file cho {split} trong {domain_dir}\n'
        f'Hãy chạy extract_domain.py trước!')


def load_embedding_matrix(dataset):
    sem_dir = os.path.join(dataset_dir, dataset, 'semantics')
    for f in os.listdir(sem_dir):
        if 'embedding_matrix' in f and f.endswith('.npy'):
            return np.load(os.path.join(sem_dir, f))
    raise FileNotFoundError(f'Không tìm thấy embedding_matrix trong {sem_dir}')


def load_dataset_with_domain(dataset, use_emotion=True):
    """
    Load đầy đủ: semantics + emotions (tùy chọn) + domain + labels.
    Trả về: data_dict, label
      data_dict = {
        'semantics': [train, val, test],
        'emotions':  [train, val, test],  # nếu use_emotion=True
        'domain':    [train, val, test],
      }
    """
    print(f'\n[Load] Dataset: {dataset} | emotion={use_emotion} | domain_topics={DOMAIN_TOPICS}')

    # Labels
    label_dir = os.path.join(dataset_dir, dataset, 'labels')
    labels = {}
    for f in os.listdir(label_dir):
        fpath = os.path.join(label_dir, f)
        if not f.endswith('.npy'):
            continue
        if   'train_' in f: labels['train'] = np.load(fpath)
        elif 'val_'   in f: labels['val']   = np.load(fpath)
        elif 'test_'  in f: labels['test']  = np.load(fpath)

    data = {'semantics': {}, 'domain': {}}
    if use_emotion:
        data['emotions'] = {}

    for split in ['train', 'val', 'test']:
        # Semantics
        sem_dir = os.path.join(dataset_dir, dataset, 'semantics')
        data['semantics'][split] = load_npy_split(sem_dir, split)

        # Emotions
        if use_emotion:
            emo_dir = os.path.join(dataset_dir, dataset, 'emotions')
            data['emotions'][split] = load_npy_split(emo_dir, split)

        # Domain
        data['domain'][split] = load_domain_split(dataset, split)

    # In shape để kiểm tra
    print()
    for split in ['train', 'val', 'test']:
        print(f'{split}:')
        print(f'  semantics : {data["semantics"][split].shape}')
        if use_emotion:
            print(f'  emotions  : {data["emotions"][split].shape}')
        print(f'  domain    : {data["domain"][split].shape}')
        print(f'  label     : {labels[split].shape}')
    print()

    embedding_matrix = load_embedding_matrix(dataset)
    return data, labels, embedding_matrix


def calculate_balanced_sample_weights(train_label):
    weights     = np.ones(len(train_label))
    label_sizes = {}
    for i, name in enumerate(labels_names[:train_label.shape[-1]]):
        arr = train_label[:, i]
        sz  = len(arr[arr == 1])
        label_sizes[name] = sz
    min_size = min(label_sizes.values())
    for lbl, size in label_sizes.items():
        index = labels_names.index(lbl)
        weights[train_label.argmax(axis=1) == index] = size / min_size
    return weights


# ─── Train ────────────────────────────────────────────────────────────────────

def train(model, dataset, data, labels, model_name,
          epochs=50, batch_size=32, use_sample_weights=False,
          emotion_dim=0, domain_dim=0):
    """
    data: dict với keys 'semantics', 'emotions' (optional), 'domain'
    labels: dict với keys 'train', 'val', 'test'
    """
    print(f'\n{"─"*20} Train {"─"*20}')

    # Chuẩn bị input theo thứ tự model expect
    # Thứ tự input: [semantics, domain (nếu có), emotions (nếu có)]
    # (phải khớp với thứ tự inputs trong build())
    def make_inputs(split):
        inputs = [data['semantics'][split]]
        if domain_dim > 0:
            inputs.append(data['domain'][split])
        if emotion_dim > 0:
            inputs.append(data['emotions'][split])
        return inputs

    train_x = make_inputs('train')
    val_x   = make_inputs('val')
    test_x  = make_inputs('test')

    train_y = labels['train']
    val_y   = labels['val']
    test_y  = labels['test']

    # Thư mục và file kết quả
    run_dir  = os.path.join(results_dir, dataset)
    os.makedirs(run_dir, exist_ok=True)
    run_name = f'{model_name}_emo{EMOTION_VERSION}_domain{DOMAIN_TOPICS}'
    out_file  = os.path.join(run_dir, f'{run_name}.json')
    hdf5_file = os.path.join(run_dir, f'{run_name}.hdf5')

    # Lưu run info
    run_info = {
        'dataset':         dataset,
        'model':           model_name,
        'emotion_version': EMOTION_VERSION,
        'emotion_dim':     emotion_dim,
        'domain_dim':      domain_dim,
        'domain_topics':   DOMAIN_TOPICS,
        'epochs_max':      epochs,
        'batch_size':      batch_size,
        'started_at':      time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    epoch_logger = EpochHistoryLogger()
    early_stop   = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint   = ModelCheckpoint(hdf5_file, monitor='val_loss',
                                   save_best_only=True, save_weights_only=True)

    sample_weights = calculate_balanced_sample_weights(train_y) \
                     if use_sample_weights else None

    t0      = time.time()
    history = model.fit(
        train_x, train_y,
        epochs=epochs, batch_size=batch_size,
        sample_weight=sample_weights,
        validation_data=(val_x, val_y),
        shuffle=True,
        callbacks=[checkpoint, early_stop, epoch_logger])
    train_time = round(time.time() - t0, 2)

    model.load_weights(hdf5_file)
    split_results = {}

    for split, x, y in [('val', val_x, val_y), ('test', test_x, test_y)]:
        print(f'\n{"─"*20} {split} {"─"*20}')
        t0_pred          = time.time()
        y_pred           = model.predict(x)
        predict_time     = round(time.time() - t0_pred, 2)
        accuracy, report = predict_single_output(y_pred, y)

        split_results[split] = {
            'accuracy':              accuracy,
            'macro_f1':              report['macro avg']['f1-score'],
            'classification_report': report,
            'predict_time_sec':      predict_time,
        }
        if dataset in datasets_en:
            split_results[split]['RMSE'] = calculate_RMSE_of_RumourEval(y_pred, y)

    result = {
        **run_info,
        'epochs_actual': len(history.history['loss']),
        'finished_at':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_time_sec': train_time,
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_acc':  float(max(history.history.get('val_accuracy', [0]))),
        'epoch_history': epoch_logger.history,
        'val':           split_results['val'],
        'test':          split_results['test'],
    }

    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'\n[Results] → {out_file}')
    print(f'[Time] Train: {train_time}s | '
          f"Val: {split_results['val']['predict_time_sec']}s | "
          f"Test: {split_results['test']['predict_time_sec']}s\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dataset, model_name, epochs=50, batch_size=32,
         l2_param=0.01, lr_param=0.001):

    CONTENT_WORDS = 100 if dataset in datasets_ch else 50

    if model_name == model_names[0]:        # DomainBiGRU (không emotion)
        data, labels, emb = load_dataset_with_domain(dataset, use_emotion=False)
        domain_dim = data['domain']['train'].shape[-1]
        model = DomainAttentionBiGRU(
            max_sequence_length=CONTENT_WORDS,
            embedding_matrix=emb,
            emotion_dim=0,
            domain_dim=domain_dim,
            category_num=labels['train'].shape[-1],
            l2_param=l2_param, lr_param=lr_param).model

    elif model_name == model_names[1]:      # DomainEmoBiGRU
        data, labels, emb = load_dataset_with_domain(dataset, use_emotion=True)
        emotion_dim = data['emotions']['train'].shape[-1]
        domain_dim  = data['domain']['train'].shape[-1]
        model = DomainAttentionBiGRU(
            max_sequence_length=CONTENT_WORDS,
            embedding_matrix=emb,
            emotion_dim=emotion_dim,
            domain_dim=domain_dim,
            category_num=labels['train'].shape[-1],
            l2_param=l2_param, lr_param=lr_param).model

    elif model_name == model_names[2]:      # DomainCNN
        data, labels, emb = load_dataset_with_domain(dataset, use_emotion=False)
        domain_dim = data['domain']['train'].shape[-1]
        model = DomainAttentionCNN(
            max_sequence_length=CONTENT_WORDS,
            embedding_matrix=emb,
            emotion_dim=0,
            domain_dim=domain_dim,
            category_num=labels['train'].shape[-1],
            l2_param=l2_param, lr_param=lr_param).model

    else:                                   # DomainEmoCNN
        data, labels, emb = load_dataset_with_domain(dataset, use_emotion=True)
        emotion_dim = data['emotions']['train'].shape[-1]
        domain_dim  = data['domain']['train'].shape[-1]
        model = DomainAttentionCNN(
            max_sequence_length=CONTENT_WORDS,
            embedding_matrix=emb,
            emotion_dim=emotion_dim,
            domain_dim=domain_dim,
            category_num=labels['train'].shape[-1],
            l2_param=l2_param, lr_param=lr_param).model

    print()
    print(model.summary())

    train(model=model, dataset=dataset, data=data, labels=labels,
          model_name=model_name, epochs=epochs, batch_size=batch_size,
          emotion_dim=data.get('emotions', {}).get('train', np.zeros(1)).shape[-1]
                      if 'emotions' in data else 0,
          domain_dim=data['domain']['train'].shape[-1])
