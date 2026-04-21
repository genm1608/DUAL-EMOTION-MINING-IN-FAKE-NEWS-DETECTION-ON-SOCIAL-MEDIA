# config_domain.py
# -----------------
# Cấu hình cho phiên bản domain.
# Chỉ cần chỉnh file này, không sửa code khác.

datasets_ch = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
datasets_en = ['RumourEval-19']

# Model names:
# Index 0: DomainBiGRU     - BiGRU + Domain (không emotion)
# Index 1: DomainEmoBiGRU  - BiGRU + Domain + Emotion  ← khuyến nghị
# Index 2: DomainCNN       - CNN + Domain
# Index 3: DomainEmoCNN    - CNN + Domain + Emotion     ← khuyến nghị
model_names = ['DomainBiGRU', 'DomainEmoBiGRU', 'DomainCNN', 'DomainEmoCNN']

# ── Chỉnh ở đây ──
experimental_dataset    = datasets_ch[0]   # Weibo-16
experimental_model_name = model_names[1]   # DomainEmoBiGRU

epochs     = 50
batch_size = 32
l2_param   = 0.01
lr_param   = 0.001
