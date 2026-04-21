# master_domain.py
# -----------------
# Entry point cho phiên bản domain.
# Chạy: python master_domain.py

import time
from train_domain import main
from config_domain import (experimental_dataset, experimental_model_name,
                            epochs, batch_size, l2_param, lr_param)

print('=' * 60)
print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}]')
print(f'Dataset : {experimental_dataset}')
print(f'Model   : {experimental_model_name}')
print(f'Epochs  : {epochs} | Batch: {batch_size}')
print(f'L2: {l2_param} | LR: {lr_param}')
print('=' * 60)

main(experimental_dataset, experimental_model_name,
     epochs, batch_size, l2_param, lr_param)
