import os, sys, time
import numpy as np
import glob
from keras.utils import to_categorical

# 1. THIẾT LẬP ĐƯỜNG DẪN IMPORT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from MLP import MLP5Layers
from sklearn.metrics import f1_score, accuracy_score

# 3. ĐƯỜNG DẪN DỮ LIỆU
DATA_DIR = r'C:\Study\LUAN VAN\DualEmotion2\code\preprocess\data\Weibo-16\emotions\v3'

def find_publisher_file(prefix):
    pattern = os.path.join(DATA_DIR, f"{prefix}_*scaled.npy")
    files = glob.glob(pattern)
    return files[0] if files else None

def get_publisher_bundle():
    try:
        y_train = to_categorical(np.load(os.path.join(DATA_DIR, "train_labels.npy")), 2)
        y_val   = to_categorical(np.load(os.path.join(DATA_DIR, "val_labels.npy")), 2)
        y_test  = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

        x_train_full = np.load(find_publisher_file("train"))
        x_val_full   = np.load(find_publisher_file("val"))
        x_test_full  = np.load(find_publisher_file("test"))

        # Trích xuất 61 chiều Publisher
        return x_train_full[:, 0:61], y_train, x_val_full[:, 0:61], y_val, x_test_full[:, 0:61], y_test
    except Exception as e:
        print(f"  [LỖI NẠP DATA] {e}")
        return None

def run_table4_v3():
    # Định nghĩa các khoảng index cần loại bỏ (Ablation)
    ablation_scenarios = [
        {"remove": "Emotion Category", "indices": slice(0, 14)},
        {"remove": "Emotion Lexicon", "indices": slice(14, 35)},
        {"remove": "Emotional Intensity", "indices": slice(35, 43)},
        {"remove": "Sentiment Score", "indices": slice(43, 44)},
        {"remove": "Other Auxiliary Features", "indices": slice(44, 61)},
        {"remove": "None (Full Publisher)", "indices": None}
    ]

    print("="*75)
    print("   TABLE 4: ABLATION STUDY (V3) — KERAS MLP | Publisher Only")
    print("="*75)

    data = get_publisher_bundle()
    if not data: return
    x_tr_raw, y_tr, x_va_raw, y_va, x_te_raw, y_te = data

    final_results = []

    for sc in ablation_scenarios:
        print(f"\n>>> Thực nghiệm: Loại bỏ {sc['remove']}")
        
        # Copy từ mảng gốc để không bị dính dữ liệu vòng lặp trước
        x_train = x_tr_raw.copy()
        x_val   = x_va_raw.copy()
        x_test  = x_te_raw.copy()
        
        if sc['indices']:
            x_train[:, sc['indices']] = 0
            x_val[:, sc['indices']] = 0
            x_test[:, sc['indices']] = 0

        # Khởi tạo model từ MLP.py của bạn
        mlp_wrapper = MLP5Layers(input_dim=61)
        model = mlp_wrapper.model 

        model.fit(x_train, y_tr, validation_data=(x_val, y_va), 
                  epochs=50, batch_size=32, verbose=0)

        # Đánh giá
        y_pred = np.argmax(model.predict(x_test), axis=1)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='macro')
        
        res_str = f"Removed: {sc['remove']:<25} | Acc: {acc:.4f} | F1: {f1:.4f}"
        print(f"  -> {res_str}")
        final_results.append(res_str)

    print("\n" + "="*75 + "\nKẾT QUẢ TỔNG HỢP TABLE 4\n" + "-"*75)
    for res in final_results: print(res)
    print("="*75)

if __name__ == "__main__":
    run_table4_v3()