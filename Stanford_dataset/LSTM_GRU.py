# ============================================================
# IMDB Sentiment Analysis with LSTM vs GRU
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
    LSTM,
    GRU,
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
# LSTM MODELİ
# ============================================================

print("\n==============================")
print("LSTM MODEL EĞİTİLİYOR")
print("==============================")

lstm_model = Sequential([
    
    Embedding(input_dim=max_words,
              output_dim=128,
              input_length=max_len),

    LSTM(64),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()

history_lstm = lstm_model.fit(
    X_train,
    train_labels,
    epochs=8,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# GRU MODELİ
# ============================================================

print("\n==============================")
print("GRU MODEL EĞİTİLİYOR")
print("==============================")

gru_model = Sequential([
    
    Embedding(input_dim=max_words,
              output_dim=128,
              input_length=max_len),

    GRU(64),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

gru_model.summary()

history_gru = gru_model.fit(
    X_train,
    train_labels,
    epochs=8,
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

# LSTM
lstm_pred_prob = lstm_model.predict(X_test)
lstm_pred = (lstm_pred_prob > 0.5).astype(int)

lstm_acc = accuracy_score(test_labels, lstm_pred)

print(f"\nLSTM Accuracy: {lstm_acc:.4f}")

# GRU
gru_pred_prob = gru_model.predict(X_test)
gru_pred = (gru_pred_prob > 0.5).astype(int)

gru_acc = accuracy_score(test_labels, gru_pred)

print(f"GRU Accuracy : {gru_acc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("LSTM Classification Report")
print("==============================")

print(classification_report(test_labels, lstm_pred))

print("\n==============================")
print("GRU Classification Report")
print("==============================")

print(classification_report(test_labels, gru_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LSTM
cm_lstm = confusion_matrix(test_labels, lstm_pred)

sns.heatmap(
    cm_lstm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0]
)

axes[0].set_title("LSTM Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# GRU
cm_gru = confusion_matrix(test_labels, gru_pred)

sns.heatmap(
    cm_gru,
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=axes[1]
)

axes[1].set_title("GRU Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# ACCURACY GRAFİĞİ
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history_lstm.history['accuracy'],
    label='LSTM Train Accuracy'
)

plt.plot(
    history_lstm.history['val_accuracy'],
    label='LSTM Val Accuracy'
)

plt.plot(
    history_gru.history['accuracy'],
    label='GRU Train Accuracy'
)

plt.plot(
    history_gru.history['val_accuracy'],
    label='GRU Val Accuracy'
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# ============================================================
# LOSS GRAFİĞİ
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history_lstm.history['loss'],
    label='LSTM Train Loss'
)

plt.plot(
    history_lstm.history['val_loss'],
    label='LSTM Val Loss'
)

plt.plot(
    history_gru.history['loss'],
    label='GRU Train Loss'
)

plt.plot(
    history_gru.history['val_loss'],
    label='GRU Val Loss'
)

plt.title("Model Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# ============================================================
# SONUÇ KARŞILAŞTIRMASI
# ============================================================

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(f"LSTM Accuracy : {lstm_acc:.4f}")
print(f"GRU Accuracy  : {gru_acc:.4f}")

if lstm_acc > gru_acc:
    print("\nDaha başarılı model: LSTM")

elif gru_acc > lstm_acc:
    print("\nDaha başarılı model: GRU")

else:
    print("\nİki modelin başarımı eşit.")


# ============================================================
# ÖRNEK TAHMİN
# ============================================================

sample_text = """
This movie was absolutely amazing.
The acting and storyline were fantastic.
"""

sample_seq = tokenizer.texts_to_sequences([sample_text])
sample_pad = pad_sequences(sample_seq, maxlen=max_len)

prediction = lstm_model.predict(sample_pad)[0][0]

print("\n==============================")
print("ÖRNEK TAHMİN")
print("==============================")

print("Metin:", sample_text)

if prediction > 0.5:
    print("Tahmin: Positive Review")
else:
    print("Tahmin: Negative Review")

print("Pozitiflik skoru:", prediction)