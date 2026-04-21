import json
import numpy as np
import os

# Cấu hình đường dẫn
JSON_DIR = r'C:\Study\LUAN VAN\DualEmotion2\dataset\Weibo-20'
SAVE_DIR = r'C:\Study\LUAN VAN\DualEmotion2\code\preprocess\data\Weibo-20\emotions\v1'

def extract_labels(file_name):
    path = os.path.join(JSON_DIR, file_name)
    labels = []
    print(f"--- Đang đọc file: {file_name} ---")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            # Chuyển đổi nhãn từ chữ sang số
            label_text = str(item['label']).lower().strip()
            
            if label_text == 'fake':
                labels.append(1)
            elif label_text == 'real':
                labels.append(0)
            else:
                # Dự phòng cho các định dạng nhãn khác như 'rumor'/'non-rumor'
                if 'fake' in label_text or 'rumor' in label_text:
                    labels.append(1)
                else:
                    labels.append(0)
                    
    return np.array(labels)

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Đã tạo thư mục lưu trữ: {SAVE_DIR}")

    # 1. Thực hiện trích xuất
    try:
        train_labels = extract_labels('train.json')
        val_labels   = extract_labels('val.json')
        test_labels  = extract_labels('test.json')

        # 2. Kiểm tra số lượng mẫu
        print("\n" + "="*30)
        print("KẾT QUẢ TRÍCH XUẤT NHÃN")
        print("-" * 30)
        print(f"Train samples: {len(train_labels):>5} (Yêu cầu: 2211)")
        print(f"Val samples:   {len(val_labels):>5} (Yêu cầu: 738)")
        print(f"Test samples:  {len(test_labels):>5} (Yêu cầu: 757)")
        print("="*30)

        # 3. Lưu file .npy
        np.save(os.path.join(SAVE_DIR, 'train_labels.npy'), train_labels)
        np.save(os.path.join(SAVE_DIR, 'val_labels.npy'), val_labels)
        np.save(os.path.join(SAVE_DIR, 'test_labels.npy'), test_labels)
        
        print(f"\n>>> THÀNH CÔNG! Đã lưu 3 file labels.npy vào thư mục v3.")
        print("Bây giờ bạn có thể chạy: python train_table3.py")

    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file JSON. Kiểm tra lại đường dẫn JSON_DIR.\n{e}")
    except Exception as e:
        print(f"LỖI KHÔNG XÁC ĐỊNH: {e}")

if __name__ == "__main__":
    main()