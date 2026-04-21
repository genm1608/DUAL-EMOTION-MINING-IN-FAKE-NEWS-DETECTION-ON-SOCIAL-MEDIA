from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant

class EmotionEnhancedCNN:
    def __init__(self, max_sequence_length, embedding_matrix, emotion_dim=0, category_num=2, hidden_units=128, l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.emotion_dim = emotion_dim
        self.hidden_units = hidden_units
        self.category_num = category_num
        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param), metrics=['accuracy'])

    def build(self):
        # --- Luồng xử lý văn bản ---
        semantic_input = Input(shape=(self.max_sequence_length,), name='word-input')
        emb = Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1],
                        embeddings_initializer=Constant(self.embedding_matrix),
                        input_length=self.max_sequence_length, trainable=False)(semantic_input)
        
        # Dùng Conv1D để bắt đặc trưng cục bộ (n-grams)
        conv = Conv1D(self.hidden_units, 5, activation='relu')(emb)
        pool = GlobalMaxPooling1D()(conv)

        # --- Luồng xử lý cảm xúc (Z-Score) ---
        if self.emotion_dim != 0:
            emotion_input = Input(shape=(self.emotion_dim,), name='emotion-input')
            # Ghép nối Text và Emotion
            merged = Concatenate()([pool, emotion_input])
            
            dense = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param))(merged)
            output = Dense(self.category_num, activation='softmax')(dense)
            model = Model(inputs=[semantic_input, emotion_input], outputs=output)
        else:
            dense = Dense(32, activation='relu')(pool)
            output = Dense(self.category_num, activation='softmax')(dense)
            model = Model(inputs=[semantic_input], outputs=output)
        return model