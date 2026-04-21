# All the datasets
datasets_ch = ['Weibo-16', 'Weibo-16-original',
               'Weibo-20', 'Weibo-20-temporal']
datasets_en = ['RumourEval-19']

# All the models
# Index: 0=MLP, 1=BiGRU, 2=EmotionEnhancedBiGRU, 3=CNN, 4=EmotionEnhancedCNN, 5=EmotionEnhancedBiLSTM
model_names = [
    'MLP', 
    'BiGRU', 
    'EmotionEnhancedBiGRU', 
    'CNN', 
    'EmotionEnhancedCNN', 
    'EmotionEnhancedBiLSTM' # Thêm vào đây
]

# Choose the following parameters of the experiments
experimental_dataset = datasets_ch[0]
experimental_model_name = model_names[2]

epochs = 50
batch_size = 32

l2_param = 0.01
lr_param = 0.001