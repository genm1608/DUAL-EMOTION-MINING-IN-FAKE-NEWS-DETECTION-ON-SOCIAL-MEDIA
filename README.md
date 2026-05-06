\## Datasets

\### Weibo-16

\### Weibo-20







\## Code



\*\*Step 1: Preprocess\*\*

```bash

cd code/preprocess

$env:EMOTION\_VERSION = "v1" (v2,3,4,... tùy điều chỉnh)

python output\_of\_labels.py

python input\_of\_emotions.py

python input\_of\_semantics.py

python normalize\_data.py

```





\*\*Step 2: Training \& Testing\*\*

```bash

cd code/train

$env:EMOTION\_VERSION = "v1" (v2,3,4,... tùy điều chỉnh)

python train\_v.py



```



Kết quả được lưu tại `train/results/`.





