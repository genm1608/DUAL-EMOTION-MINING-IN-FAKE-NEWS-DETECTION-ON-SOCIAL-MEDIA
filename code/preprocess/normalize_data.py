import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_emotions(dataset_name, base_dir='./data'):
    EMOTION_VER = os.environ.get('EMOTION_VERSION', 'v1')
    
    # 1. Đường dẫn gốc tới thư mục emotions
    emotion_base_dir = os.path.join(base_dir, dataset_name, 'emotions')
    
    # 2. ĐƯỜNG DẪN NGUỒN (Source): Phải trỏ vào thư mục VERSION cụ thể
    # Ví dụ: data/Weibo-16/emotions/v2
    source_dir = os.path.join(emotion_base_dir, EMOTION_VER)
    
    if not os.path.exists(source_dir):
        print(f"  -> Lỗi: Không tìm thấy dữ liệu thô tại {source_dir}")
        return
        
    print(f"\n[Standardize] Đang chuẩn hóa cho {dataset_name} | Phiên bản: {EMOTION_VER}")
    
    # Quét file trong thư mục của Version (source_dir)
    files = os.listdir(source_dir)
    
    try:
        # Tìm file gốc (chưa có chữ _scaled) ngay trong thư mục v2, v5, v8...
        train_file = [f for f in files if f.startswith('train_') and '_scaled' not in f][0]
        val_file   = [f for f in files if f.startswith('val_')   and '_scaled' not in f][0]
        test_file  = [f for f in files if f.startswith('test_')  and '_scaled' not in f][0]
        
        print(f"  -> File gốc tìm thấy: {train_file}") # Sẽ hiện đúng số (..., 290) hoặc (..., 380)
    except IndexError:
        print(f"  -> Lỗi: Thư mục {source_dir} không chứa file .npy thô.")
        return

    # 3. Load dữ liệu từ SOURCE_DIR (thư mục version)
    train_arr = np.load(os.path.join(source_dir, train_file))
    val_arr   = np.load(os.path.join(source_dir, val_file))
    test_arr  = np.load(os.path.join(source_dir, test_file))

    # 4. Chuẩn hóa
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    val_scaled   = scaler.transform(val_arr)
    test_scaled  = scaler.transform(test_arr)

    # 5. Lưu đè bản _scaled vào cùng thư mục SOURCE_DIR
    np.save(os.path.join(source_dir, train_file.replace('.npy', '_scaled.npy')), train_scaled)
    np.save(os.path.join(source_dir, val_file.replace('.npy', '_scaled.npy')), val_scaled)
    np.save(os.path.join(source_dir, test_file.replace('.npy', '_scaled.npy')), test_scaled)
    
    print(f"  -> Thành công! Đã lưu bản chuẩn hóa vào: {source_dir}")

if __name__ == '__main__':
    datasets = ['Weibo-16', 'Weibo-20', 'Weibo-20-temporal']
    for ds in datasets:
        scale_emotions(ds)