import os
import json
import math
import sys
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

sys.path.append('../model')
from MLP import MLP5Layers
from BiGRU import EmotionEnhancedBiGRU
from CNN import CNNModel
from config import model_names


# ─── Constants ────────────────────────────────────────────────────────────────

labels_names = ['fake', 'real', 'unverified']
datasets_ch  = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
datasets_en  = ['RumourEval-19']

dataset_dir  = '../preprocess/data'
results_dir  = './results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

EMOTION_VERSION = os.environ.get('EMOTION_VERSION', 'v1')


# ─── Callback: lưu lịch sử từng epoch ────────────────────────────────────────

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

    print(f'\nTEST_sz: {len(test_label)}')
    for i, name in enumerate(names):
        arr = test_label[:, i]
        print(f'{name}_sz: {len(arr[arr == 1])}')
    print(f'\nAccuracy: {accuracy:.3f}\n')
    print(classification_report(test_label, y_pred_label,
                                labels=list(range(len(names))),
                                target_names=names, digits=3))
    report = classification_report(test_label, y_pred_label,
                                   labels=list(range(len(names))),
                                   target_names=names, digits=3, output_dict=True)
    return accuracy, report


# ─── Load dataset ─────────────────────────────────────────────────────────────

def load_dataset(dataset, input_types=['emotions']):
    assert dataset in datasets_ch + datasets_en
    for t in input_types:
        assert t in ['emotions', 'semantics']

    label_dir = os.path.join(dataset_dir, dataset, 'labels')
    train_label = val_label = test_label = None
    for f in os.listdir(label_dir):
        fpath = os.path.join(label_dir, f)
        if   'train_' in f: train_label = np.load(fpath)
        elif 'val_'   in f: val_label   = np.load(fpath)
        elif 'test_'  in f: test_label  = np.load(fpath)

    train_data, val_data, test_data = [], [], []
    semantics_embedding_matrix      = None

    for t in input_types:
        data_dir = os.path.join(dataset_dir, dataset, t)
        _train = _val = _test = None
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if not f.endswith(".npy"):
                continue
            if   "embedding_matrix_" in f: semantics_embedding_matrix = np.load(fpath)
            elif "train_"            in f and _train is None: _train = np.load(fpath)
            elif "val_"              in f and _val   is None: _val   = np.load(fpath)
            elif "test_"             in f and _test  is None: _test  = np.load(fpath)
        train_data.append(_train)
        val_data.append(_val)
        test_data.append(_test)

    if len(input_types) == 1:
        train_data, val_data, test_data = train_data[0], val_data[0], test_data[0]

    data  = [train_data, val_data, test_data]
    label = [train_label, val_label, test_label]

    print()
    for i, t in enumerate(['Train', 'Val', 'Test']):
        if len(input_types) == 1:
            print(f'{t} data: {data[i].shape},  {t} label: {label[i].shape}')
        else:
            print(f'{t} data:')
            for j, it in enumerate(input_types):
                print(f'  [{it}]\t{data[i][j].shape}')
            print(f'  {t} label: {label[i].shape}')
    print()

    if 'semantics' in input_types:
        return data, label, semantics_embedding_matrix
    return data, label


# ─── Sample weights ───────────────────────────────────────────────────────────

def calculate_balanced_sample_weights(train_label):
    weights     = np.ones(len(train_label))
    label_sizes = {}
    print(f'\nIn train_label {train_label.shape}:')
    for i, name in enumerate(labels_names[:train_label.shape[-1]]):
        arr = train_label[:, i]
        sz  = len(arr[arr == 1])
        label_sizes[name] = sz
        print(f'{name}_sz: {sz}')
    print()
    min_size = min(label_sizes.values())
    for lbl, size in label_sizes.items():
        index = labels_names.index(lbl)
        weights[train_label.argmax(axis=1) == index] = size / min_size
    return weights


# ─── Train ────────────────────────────────────────────────────────────────────

def train(model, dataset, data, label, model_name,
          epochs=50, batch_size=32, use_sample_weights=False, emotion_dim=0):

    print(f'\n{"─"*20} Train {"─"*20}\n')

    train_data, val_data, test_data    = data
    train_label, val_label, test_label = label

    # Thư mục và file kết quả: results/<dataset>/<model>_emo<version>.json
    run_dir   = os.path.join(results_dir, dataset)
    os.makedirs(run_dir, exist_ok=True)
    run_name  = f'{model_name}_emo{EMOTION_VERSION}'
    out_file  = os.path.join(run_dir, f'{run_name}.json')
    hdf5_file = os.path.join(run_dir, f'{run_name}.hdf5')

    # ── Training ──
    epoch_logger = EpochHistoryLogger()
    early_stop   = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint   = ModelCheckpoint(hdf5_file, monitor='val_loss',
                                   save_best_only=True, save_weights_only=True)

    sample_weights = calculate_balanced_sample_weights(train_label) \
                     if use_sample_weights else None

    t0_train = time.time()
    history  = model.fit(
        train_data, train_label,
        epochs=epochs, batch_size=batch_size,
        sample_weight=sample_weights,
        validation_data=(val_data, val_label),
        shuffle=True,
        callbacks=[checkpoint, early_stop, epoch_logger])
    train_time_sec = round(time.time() - t0_train, 2)

    # ── Predict val + test ──
    model.load_weights(hdf5_file)
    split_results = {}

    for i, split in enumerate(['val', 'test']):
        print(f'\n{"─"*20} {split} {"─"*20}\n')

        t0_pred          = time.time()
        y_pred           = model.predict(data[1 + i])
        predict_time_sec = round(time.time() - t0_pred, 2)

        accuracy, report = predict_single_output(y_pred, label[1 + i])

        split_results[split] = {
            'accuracy':              accuracy,
            'macro_f1':              report['macro avg']['f1-score'],
            'classification_report': report,
            'predict_time_sec':      predict_time_sec,
        }
        if dataset in datasets_en:
            split_results[split]['RMSE'] = calculate_RMSE_of_RumourEval(
                y_pred, label[1 + i])

    # ── Ghi 1 file JSON duy nhất ──
    result = {
        'dataset':         dataset,
        'model':           model_name,
        'emotion_version': EMOTION_VERSION,
        'emotion_dim':     emotion_dim,
        'epochs_max':      epochs,
        'epochs_actual':   len(history.history['loss']),
        'batch_size':      batch_size,
        'train_time_sec':  train_time_sec,
        'best_val_loss':   float(min(history.history['val_loss'])),
        'best_val_acc':    float(max(history.history.get('val_accuracy', [0]))),
        'epoch_history':   epoch_logger.history,
        'val':             split_results['val'],
        'test':            split_results['test'],
    }

    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n[Results] → {out_file}')
    print(f'[Time] Train: {train_time_sec}s | '
          f"Val predict: {split_results['val']['predict_time_sec']}s | "
          f"Test predict: {split_results['test']['predict_time_sec']}s\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(experimental_dataset, experimental_model_name,
         epochs, batch_size, l2_param, lr_param):

    emotion_dim = 0

    if experimental_model_name == model_names[0]:           # MLP
        data, label = load_dataset(experimental_dataset, input_types=['emotions'])
        emotion_dim = data[0].shape[-1]
        model = MLP5Layers(input_dim=emotion_dim,
                           category_num=label[0].shape[-1],
                           l2_param=l2_param, lr_param=lr_param).model

    elif experimental_model_name == model_names[1]:         # BiGRU
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics'])
        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        model = EmotionEnhancedBiGRU(max_sequence_length=CONTENT_WORDS,
                                     embedding_matrix=embedding_matrix,
                                     emotion_dim=0,
                                     category_num=label[0].shape[-1],
                                     l2_param=l2_param, lr_param=lr_param).model

    elif experimental_model_name == model_names[2]:         # EmotionEnhancedBiGRU
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics', 'emotions'])
        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        emotion_dim   = data[0][1].shape[-1]
        model = EmotionEnhancedBiGRU(max_sequence_length=CONTENT_WORDS,
                                     embedding_matrix=embedding_matrix,
                                     emotion_dim=emotion_dim,
                                     category_num=label[0].shape[-1],
                                     l2_param=l2_param, lr_param=lr_param).model

    elif experimental_model_name == model_names[3]:         # CNN
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics'])
        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        model = CNNModel(max_sequence_length=CONTENT_WORDS,
                         embedding_matrix=embedding_matrix,
                         emotion_dim=0,
                         category_num=label[0].shape[-1],
                         l2_param=l2_param, lr_param=lr_param).model

    else:                                                    # EmotionEnhancedCNN
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics', 'emotions'])
        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        emotion_dim   = data[0][1].shape[-1]
        model = CNNModel(max_sequence_length=CONTENT_WORDS,
                         embedding_matrix=embedding_matrix,
                         emotion_dim=emotion_dim,
                         category_num=label[0].shape[-1],
                         l2_param=l2_param, lr_param=lr_param).model

    print()
    print(model.summary())
    print()

    train(model=model, dataset=experimental_dataset, data=data,
          label=label, model_name=experimental_model_name,
          epochs=epochs, batch_size=batch_size, emotion_dim=emotion_dim)