import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time

# sklearn & plot
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1️⃣ MNIST yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Subset
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

# 3️⃣ Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4️⃣ Resize + RGB
def preprocess(images):
    images = tf.image.resize(images[..., np.newaxis], (224, 224))
    images = tf.image.grayscale_to_rgb(images)
    return images.numpy()

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# 🔥 5️⃣ Inception modülü
def inception_module(x, f1, f3_in, f3_out, f5_in, f5_out, pool_proj):
    path1 = layers.Conv2D(f1, (1,1), activation='relu', padding='same')(x)

    path2 = layers.Conv2D(f3_in, (1,1), activation='relu', padding='same')(x)
    path2 = layers.Conv2D(f3_out, (3,3), activation='relu', padding='same')(path2)

    path3 = layers.Conv2D(f5_in, (1,1), activation='relu', padding='same')(x)
    path3 = layers.Conv2D(f5_out, (5,5), activation='relu', padding='same')(path3)

    path4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    path4 = layers.Conv2D(pool_proj, (1,1), activation='relu', padding='same')(path4)

    return layers.concatenate([path1, path2, path3, path4])

# 🔥 6️⃣ Model
input_layer = layers.Input(shape=(224,224,3))

x = layers.Conv2D(64, (7,7), strides=2, activation='relu', padding='same')(input_layer)
x = layers.MaxPooling2D((3,3), strides=2)(x)

x = layers.Conv2D(64, (1,1), activation='relu')(x)
x = layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((3,3), strides=2)(x)

x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)

x = layers.MaxPooling2D((3,3), strides=2)(x)

x = inception_module(x, 192, 96, 208, 16, 48, 64)
x = inception_module(x, 160, 112, 224, 24, 64, 64)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)
output = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output)

# 7️⃣ Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🔥 8️⃣ Train (time ölç)
start = time.time()

history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test)
)

end = time.time()
train_time = end - start

# 9️⃣ Test
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# 🔟 Tahmin
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 1️⃣1️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("GoogLeNet Confusion Matrix")

plt.savefig("googlenet_confusion_matrix.png")
plt.show()

# 1️⃣2️⃣ Classification Report
report = classification_report(y_test, y_pred_classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("googlenet_classification_report.csv")

# 1️⃣3️⃣ Tahmin kayıt
df_preds = pd.DataFrame({
    "Gerçek": y_test,
    "Tahmin": y_pred_classes
})
df_preds.to_csv("googlenet_predictions.csv", index=False)

# 1️⃣4️⃣ SUMMARY TABLO (EN ÖNEMLİ)
df_summary = pd.DataFrame([{
    "Model": "GoogLeNet",
    "Accuracy": test_acc,
    "Precision": report["weighted avg"]["precision"],
    "Recall": report["weighted avg"]["recall"],
    "F1-score": report["weighted avg"]["f1-score"],
    "Epoch": 3,
    "Batch Size": 64,
    "Train Time": train_time
}])

df_summary.to_csv(
    "model_summary.csv",
    mode='a',
    index=False,
    header=not os.path.exists("model_summary.csv")
)

print("✅ Tüm çıktılar kaydedildi")