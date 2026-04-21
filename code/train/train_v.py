"""
train_v_final.py — ABLATION STUDY theo paper gốc (Zhang et al., WWW 2021)
6 kịch bản × 3 mô hình (BiGRU, BiLSTM, CNN):
  1. Semantic only
  2. Semantic + EmoRatio   
  3. Semantic + EmoCred    
  4. Semantic + Dual Emotion full 


Cách chạy: set EMOTION_VERSION=v1 && python train_v_final.py
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
from BiGRU import EmotionEnhancedBiGRU
from BiLSTM import EmotionEnhancedBiLSTM
from CNN import EmotionEnhancedCNN
from MLP import MLP5Layers

DATASET       = 'Weibo-20'
# ['Weibo-16', 'Weibo-20', 'Weibo-20-temporal']
EMOTION_VER   = os.environ.get('EMOTION_VERSION', 'v1')
EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 0.001
L2            = 0.01
CONTENT_WORDS = 100
PATIENCE      = 10

PREPROCESS_DIR = os.path.join('..', '..', 'code', 'preprocess', 'data', DATASET)
EMO_DIR        = os.path.join(PREPROCESS_DIR, 'emotions')
OUT_DIR        = os.path.join('results', DATASET, EMOTION_VER + '_Ablation')
os.makedirs(OUT_DIR, exist_ok=True)


def find_npy(folder, keys):
    if not os.path.exists(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if f.endswith('.npy') and all(k in f for k in keys):
            return np.load(os.path.join(folder, f))
    return None


def load_all():
    sem_dir = os.path.join(PREPROCESS_DIR, 'semantics')
    lbl_dir = os.path.join(PREPROCESS_DIR, 'labels')
    ver_dir = os.path.join(EMO_DIR, EMOTION_VER)
    splits  = {}

    for sp in ['train', 'val', 'test']:
        # 1. Tìm file dual (giữ nguyên logic đã sửa trước đó)
        dual = find_npy(ver_dir, [sp, '_scaled'])
        if dual is None:
            dual = find_npy(EMO_DIR, [sp, EMOTION_VER, '_scaled'])
        if dual is None:
            dual = find_npy(ver_dir, [sp])
        if dual is None:
            dual = find_npy(EMO_DIR, [sp, EMOTION_VER])

        # 2. Tìm file emocred (16d) và emoratio (1d)
        emocred  = find_npy(ver_dir, [sp, 'emocred'])
        emoratio = find_npy(ver_dir, [sp, 'emoratio'])

        # 3. Gán vào từ điển splits theo khóa sp ('train', 'val', hoặc 'test')
        splits[sp] = {
            'sem':      find_npy(sem_dir, [sp]),
            'lbl':      find_npy(lbl_dir, [sp]),
            'dual':     dual,
            'emocred':  emocred,
            'emoratio': emoratio,
        }
        
        # In log để kiểm tra tiến trình nạp từng tập dữ liệu
        print('  [{}] dual={} | emocred={} | emoratio={}'.format(
            sp,
            dual.shape     if dual     is not None else None,
            emocred.shape  if emocred  is not None else None,
            emoratio.shape if emoratio is not None else None))

    # 4. Tìm file Embedding (Nạp bên ngoài vòng lặp for)
    embed = find_npy(sem_dir, ['embed'])
    if embed is None:
        fallback_path = os.path.join(PREPROCESS_DIR, '..', 'word-embedding')
        embed = find_npy(fallback_path, [])
        
    return splits, embed


class EpochHistory(Callback):
    def __init__(self):
        super().__init__()
        self.records = []
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.records.append({
            'epoch':      epoch + 1,
            'train_loss': float(logs.get('loss', 0)),
            'train_acc':  float(logs.get('accuracy', 0)),
            'val_loss':   float(logs.get('val_loss', 0)),
            'val_acc':    float(logs.get('val_accuracy', 0)),
            'epoch_time': round(time.time() - self._t0, 2),
        })


def visualize(records, y_true, y_pred, tag, tag_safe):
    ep = [r['epoch']      for r in records]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, [r['train_loss'] for r in records], 'b-', label='Train')
    ax1.plot(ep, [r['val_loss']   for r in records], 'r-', label='Val')
    ax1.set_title('Loss\n' + tag); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.plot(ep, [r['train_acc'] for r in records], 'g-',     label='Train')
    ax2.plot(ep, [r['val_acc']   for r in records], color='orange', label='Val')
    ax2.set_title('Accuracy\n' + tag); ax2.set_ylim([0.5,1.0])
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot_' + tag_safe + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_title('CM\n' + tag)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Real','Fake'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Real','Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    thresh   = cm.max() / 2
    lbl_map  = {(0,0):'TN',(0,1):'FP',(1,0):'FN',(1,1):'TP'}
    for i in range(2):
        for j in range(2):
            c = 'white' if cm[i,j] > thresh else 'black'
            ax.text(j, i,      str(cm[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=c)
            ax.text(j, i+0.30, '('+lbl_map[(i,j)]+')',
                    ha='center', va='center', fontsize=9, color=c)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cm_' + tag_safe + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def run_experiment(tag, tag_safe, model, train_x, val_x, test_x,
                   train_y, val_y, test_y):
    n_test = test_y.shape[0]
    print('\n[START] ' + tag)

    # early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE,
    #                            restore_best_weights=True, verbose=0)
    cb = EpochHistory()
    t0 = time.time()
    model.fit(train_x, train_y, validation_data=(val_x, val_y),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[cb], verbose=0) # Chỉ còn giữ lại [cb]
              
    t_train = time.time() - t0
    stopped = len(cb.records)

    t1     = time.time()
    y_prob = model.predict(test_x, batch_size=BATCH_SIZE, verbose=0)
    t_test = time.time() - t1

    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(test_y, axis=1)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true,  y_pred, average='macro')
    prec   = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec    = recall_score(y_true,    y_pred, average='macro', zero_division=0)

    print('\n' + '-'*55)
    print('  ' + tag)
    print('-'*55)
    print('  Accuracy   : {:.3f}'.format(acc))
    print('  Macro F1   : {:.3f}'.format(f1))
    print('  Precision  : {:.3f}'.format(prec))
    print('  Recall     : {:.3f}'.format(rec))
    print('  Epoch stop : {}/{}'.format(stopped, EPOCHS))
    print('  Train time : {:.3f}s'.format(t_train))
    print('  Test  time : {:.3f}s  ({:.3f}ms/sample)'.format(
          t_test, t_test/n_test*1000))
    print('-'*55)
    print(classification_report(y_true, y_pred,
          target_names=['Real','Fake'], digits=3, zero_division=0))

    cm = visualize(cb.records, y_true, y_pred, tag, tag_safe)
    return {
        'tag': tag, 'model': '',
        'acc': round(acc,3), 'f1': round(f1,3),
        'precision': round(prec,3), 'recall': round(rec,3),
        'epoch_stopped': stopped,
        'train_time_s': round(t_train,3), 'test_time_s': round(t_test,3),
        'ms_per_sample': round(t_test/n_test*1000,3),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=['Real','Fake'],
            output_dict=True, zero_division=0),
        'epoch_history': cb.records,
    }


def main():
    print('='*60)
    print('  DATASET : ' + DATASET + ' | VER : ' + EMOTION_VER)
    print('  EPOCHS  : {} | PATIENCE : {}'.format(EPOCHS, PATIENCE))
    print('='*60)

    splits, embed = load_all()

    for sp in ['train','val','test']:
        if splits[sp]['sem'] is None:
            print('[ERROR] Missing sem/' + sp); return
        if splits[sp]['lbl'] is None:
            print('[ERROR] Missing lbl/' + sp); return

    dual_dim     = splits['train']['dual'].shape[-1]     if splits['train']['dual']     is not None else 0
    emocred_dim  = splits['train']['emocred'].shape[-1]  if splits['train']['emocred']  is not None else 0
    emoratio_dim = splits['train']['emoratio'].shape[-1] if splits['train']['emoratio'] is not None else 0

    def s(sp): return splits[sp]['sem']
    def l(sp): return splits[sp]['lbl']
    def d(sp): return splits[sp]['dual']
    def c(sp): return splits[sp]['emocred']
    def r(sp): return splits[sp]['emoratio']

    all_res = []
    models  = [
        (EmotionEnhancedBiGRU,  'BiGRU',  'bigru'),
        (EmotionEnhancedBiLSTM, 'BiLSTM', 'bilstm'),
        (EmotionEnhancedCNN,    'CNN',    'cnn'),
    ]

    for Cls, mname, mtag in models:
        print('\n' + '='*60 + '\n  MODEL: ' + mname + '\n' + '='*60)

        def run(no, label, tag_s, x_tr, x_v, x_te, dim=0, mlp=False):
            K.clear_session()
            if mlp:
                m = MLP5Layers(input_dim=dim, category_num=2,
                               l2_param=L2, lr_param=LR).model
            else:
                m = Cls(CONTENT_WORDS, embed, emotion_dim=dim,
                        l2_param=L2, lr_param=LR).model
            res = run_experiment(
                '{} | ({}) {}'.format(mname, no, label),
                '{}_{}'.format(mtag, tag_s),
                m, x_tr, x_v, x_te,
                l('train'), l('val'), l('test'))
            res['model'] = mname
            all_res.append(res)

        # 1. Semantic only
        run(1, 'Semantic only', '1_sem',
            s('train'), s('val'), s('test'), dim=0)

        # 2. Semantic + EmoRatio
        if emoratio_dim > 0:
            run(2, 'Sem + EmoRatio ({}d)'.format(emoratio_dim), '2_emoratio',
                [s('train'),r('train')], [s('val'),r('val')], [s('test'),r('test')],
                dim=emoratio_dim)

        # 3. Semantic + EmoCred
        if emocred_dim > 0:
            run(3, 'Sem + EmoCred ({}d)'.format(emocred_dim), '3_emocred',
                [s('train'),c('train')], [s('val'),c('val')], [s('test'),c('test')],
                dim=emocred_dim)

        # 4. Semantic + Dual Emotion full
        if dual_dim > 0:
            run(4, 'Sem + DualEmotion {} ({}d)'.format(EMOTION_VER, dual_dim),
                '4_dual',
                [s('train'),d('train')], [s('val'),d('val')], [s('test'),d('test')],
                dim=dual_dim)

        # # 5. EmoRatio only MLP
        # if emoratio_dim > 0:
        #     run(5, 'EmoRatio only MLP', '5_emoratio_only',
        #         r('train'), r('val'), r('test'),
        #         dim=emoratio_dim, mlp=True)

        # # 6. EmoCred only MLP
        # if emocred_dim > 0:
        #     run(6, 'EmoCred only MLP', '6_emocred_only',
        #         c('train'), c('val'), c('test'),
        #         dim=emocred_dim, mlp=True)

    # Bảng tổng kết
  
 
    sep = '=' * 105
    print('\n' + sep)
    print(('TỔNG HỢP — ' + DATASET + ' | ' + EMOTION_VER).center(105))
    print(sep)
    
    # Tiêu đề bảng: Acc, F1, Prec, Rec (7 ký tự); Train, Test (10 ký tự)
    print('{:<10} {:<36} {:>7} {:>7} {:>7} {:>7} {:>10} {:>10}'.format(
          'Model', 'Kịch bản', 'Acc', 'F1', 'Prec', 'Rec', 'Train (s)', 'Test (s)'))
    print('-' * 105)
    
    cur = ''
    for r in all_res:
        if cur and cur != r['model']:
            print('-' * 105)
        cur = r['model']
        
        # Trích xuất tên kịch bản (ví dụ: (4) Sem + DualEmotion...)
        sc = r['tag'].split('|')[-1].strip()[:36]
        
        # In dữ liệu: Train lấy 3 chữ số thập phân, Test lấy 3 chữ số thập phân
        print('{:<10} {:<36} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>10.3f} {:>10.3f}'.format(
              r['model'], 
              sc, 
              r['acc'], 
              r['f1'], 
              r['precision'], 
              r['recall'],
              r['train_time_s'], 
              r['test_time_s']))
              
    print(sep)

    # Lưu file JSON để lưu trữ toàn bộ lịch sử (bao gồm cả epoch_stopped)
    with open(os.path.join(OUT_DIR, 'ablation_' + EMOTION_VER + '.json'),
              'w', encoding='utf-8') as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print('\nSaved -> ' + OUT_DIR + '/')


if __name__ == '__main__':
    main()