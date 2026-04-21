import os, sys, time
import numpy as np
import glob
from keras.utils import to_categorical

# 1. THIẾT LẬP ĐƯỜNG DẪN IMPORT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

# 2. IMPORT MÔ HÌNH KERAS TỪ FILE CỦA BẠN
from MLP import MLP5Layers
from sklearn.metrics import f1_score, accuracy_score

# 3. ĐƯỜNG DẪN DỮ LIỆU
DATA_DIR = r'C:\Study\LUAN VAN\DualEmotion2\code\preprocess\data\Weibo-20\emotions\v1'

def find_file(prefix, feature_name):
    # Tìm tất cả file bắt đầu bằng prefix (train/val/test) và chứa tên feature
    pattern = os.path.join(DATA_DIR, f"{prefix}_{feature_name}*.npy")
    files = glob.glob(pattern)
    
    # Nếu không thấy, thử tìm kiểu chứa feature name ở giữa (cho file scaled)
    if not files:
        pattern = os.path.join(DATA_DIR, f"{prefix}_*{feature_name}*.npy")
        files = glob.glob(pattern)
        
    if files:
        return files[0]
    else:
        print(f"  [CẢNH BÁO] Không tìm thấy file cho: {prefix} + {feature_name}")
        return None

def get_data_bundle(feature_key, slice_range=None):
    try:
        # Keras categorical_crossentropy yêu cầu nhãn dạng One-hot
        y_train = to_categorical(np.load(os.path.join(DATA_DIR, "train_labels.npy")), 2)
        y_val   = to_categorical(np.load(os.path.join(DATA_DIR, "val_labels.npy")), 2)
        y_test  = np.load(os.path.join(DATA_DIR, "test_labels.npy")) # Test giữ nguyên để dùng sklearn

        target = "scaled" if slice_range else feature_key
        x_train = np.load(find_file("train", target))
        x_val   = np.load(find_file("val", target))
        x_test  = np.load(find_file("test", target))

        if slice_range:
            s, e = slice_range
            x_train, x_val, x_test = x_train[:, s:e], x_val[:, s:e], x_test[:, s:e]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    except Exception as e:
        print(f"  [LỖI NẠP DATA] {e}")
        return None

def run_table3_v3():
    scenarios = [
        {"name": "Emoratio", "key": "emoratio", "slice": None},
        {"name": "EmoCred", "key": "emocred", "slice": None},
        {"name": "Publisher Emotion", "key": "scaled", "slice": (0, 61)},
        {"name": "Social Emotion", "key": "scaled", "slice": (61, 183)},
        {"name": "Emotion Gap", "key": "scaled", "slice": (183, 305)},
        {"name": "Dual Emotion Features", "key": "scaled", "slice": (0, 305)}
    ]

    print("="*75)
    print("   TABLE 3 REPRODUCTION (V3) — KERAS MLP5Layers")
    print("="*75)

    final_results = []
    for sc in scenarios:
        print(f"\n>>> Đang xử lý kịch bản: {sc['name']}")
        bundle = get_data_bundle(sc['key'], sc['slice'])
        if not bundle: continue
        
        (x_train, y_train), (x_val, y_val), (x_test, y_test_raw) = bundle
        
        # KHỞI TẠO: Vì file MLP.py của bạn đã compile sẵn trong __init__
        # Chúng ta truy cập vào thuộc tính .model
        mlp_wrapper = MLP5Layers(input_dim=x_train.shape[1])
        model = mlp_wrapper.model 

        # HUẤN LUYỆN
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # ĐÁNH GIÁ
        y_pred_probs = model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        acc = accuracy_score(y_test_raw, y_pred)
        f1 = f1_score(y_test_raw, y_pred, average='macro')
        
        res_str = f"{sc['name']:<25} | Dim: {x_train.shape[1]:>3} | Acc: {acc:.4f} | F1: {f1:.4f}"
        print(f"  -> {res_str}")
        final_results.append(res_str)

    print("\n" + "="*75 + "\nKẾT QUẢ TỔNG HỢP TABLE 3\n" + "-"*75)
    for res in final_results: print(res)
    print("="*75)

if __name__ == "__main__":
    run_table3_v3()