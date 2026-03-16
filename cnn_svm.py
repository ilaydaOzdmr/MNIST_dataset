import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, Model, Input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("1- Program başladı")

# 1) MNIST verisini yükle
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print("2- Veri yüklendi")

# 2) Normalize et
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3) CNN için reshape
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
print("3- Veri hazırlandı:", x_train.shape, x_test.shape)

# 4) Functional API ile CNN modeli kur
inputs = Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
features = layers.Dense(128, activation='relu', name="feature_dense")(x)
outputs = layers.Dense(10, activation='softmax')(features)

cnn_model = Model(inputs=inputs, outputs=outputs)

print("4- CNN modeli oluşturuldu")

# 5) CNN modeli derle
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("5- CNN modeli derlendi")
cnn_model.summary()

# 6) CNN'i eğit
history = cnn_model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("6- CNN eğitimi tamamlandı")

# 7) Özellik çıkarıcı model
feature_extractor = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer("feature_dense").output
)

print("7- Feature extractor oluşturuldu")

# 8) Özellikleri çıkar
train_features = feature_extractor.predict(x_train, verbose=1)
test_features = feature_extractor.predict(x_test, verbose=1)

print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)

# 9) SVM eğit
# İlk deneme için tüm veri yavaş olabilir, istersen 10000 ile başla
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Hızlı deneme için:
# svm_model.fit(train_features[:10000], y_train[:10000])

svm_model.fit(train_features, y_train)
print("8- SVM eğitildi")

# 10) Tahmin
y_pred = svm_model.predict(test_features)

# 11) Sonuçlar
acc = accuracy_score(y_test, y_pred)
print("CNN + SVM Test Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("9- Program bitti")