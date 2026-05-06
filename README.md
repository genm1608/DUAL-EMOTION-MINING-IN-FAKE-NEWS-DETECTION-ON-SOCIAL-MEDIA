\## Datasets

\### Weibo-16

\### Weibo-20







\## Code



\*\*Step 1: Preprocess\*\*

```bash

cd code/preprocess

$env:EMOTION_VERSION = "v1" (v2,3,4,... tùy điều chỉnh)

python output_of_labels.py

python input_of_emotions.py

python input_of_semantics.py

python normalize_data.py

```





\*\*Step 2: Training \& Testing\*\*

```bash

cd code/train

$env:EMOTION_VERSION = "v1" (v2,3,4,... tùy điều chỉnh)

python train_v.py



```



Kết quả được lưu tại `train/results/`.





