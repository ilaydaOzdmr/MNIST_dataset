# ============================================================
# IMDB Sentiment Analysis with Simple RNN
# ============================================================

# KURULUM:
# pip install datasets tensorflow scikit-learn matplotlib seaborn

# ============================================================
# KÜTÜPHANELER
# ============================================================

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# VERİ SETİNİ YÜKLE
# ============================================================

print("IMDB veri seti yükleniyor...")

dataset = load_dataset("stanfordnlp/imdb")

train_texts = dataset["train"]["text"]
train_labels = np.array(dataset["train"]["label"])

test_texts = dataset["test"]["text"]
test_labels = np.array(dataset["test"]["label"])

print("\nTrain örnek sayısı:", len(train_texts))
print("Test örnek sayısı :", len(test_texts))


# ============================================================
# TOKENIZATION & PADDING
# ============================================================

max_words = 10000
max_len = 200

print("\nTokenization başlıyor...")

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)


# ============================================================
# SIMPLE RNN MODELİ
# ============================================================

print("\n==============================")
print("SIMPLE RNN MODEL EĞİTİLİYOR")
print("==============================")

rnn_model = Sequential([

    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_len
    ),

    SimpleRNN(64),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rnn_model.summary()


# ============================================================
# MODEL EĞİTİMİ
# ============================================================

history_rnn = rnn_model.fit(
    X_train,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# TEST SONUÇLARI
# ============================================================

print("\n==============================")
print("MODEL TEST SONUÇLARI")
print("==============================")

rnn_pred_prob = rnn_model.predict(X_test)

rnn_pred = (rnn_pred_prob > 0.5).astype(int)

rnn_acc = accuracy_score(test_labels, rnn_pred)

print(f"\nRNN Accuracy: {rnn_acc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("RNN Classification Report")
print("==============================")

print(classification_report(test_labels, rnn_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(test_labels, rnn_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("RNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================================
# ACCURACY GRAFİĞİ
# ============================================================

epochs = range(1, len(history_rnn.history['accuracy']) + 1)

plt.figure(figsize=(10,5))

plt.plot(
    epochs,
    history_rnn.history['accuracy'],
    marker='o',
    label='Train Accuracy'
)

plt.plot(
    epochs,
    history_rnn.history['val_accuracy'],
    marker='o',
    label='Validation Accuracy'
)

plt.xticks(list(epochs))

plt.title("RNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# LOSS GRAFİĞİ
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    epochs,
    history_rnn.history['loss'],
    marker='o',
    label='Train Loss'
)

plt.plot(
    epochs,
    history_rnn.history['val_loss'],
    marker='o',
    label='Validation Loss'
)

plt.xticks(list(epochs))

plt.title("RNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# ÖRNEK TAHMİN
# ============================================================

sample_text = """
This movie was fantastic.
I really loved the acting and story.
"""

sample_seq = tokenizer.texts_to_sequences([sample_text])

sample_pad = pad_sequences(
    sample_seq,
    maxlen=max_len
)

prediction = rnn_model.predict(sample_pad)[0][0]

print("\n==============================")
print("ÖRNEK TAHMİN")
print("==============================")

print("Metin:", sample_text)

if prediction > 0.5:
    print("Tahmin: Positive Review")
else:
    print("Tahmin: Negative Review")

print("Pozitiflik skoru:", prediction)