# import tensorflow as tf
# from tensorflow.keras import layers, models
# import numpy as np

# # 1️⃣ MNIST yükle
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # 2️⃣ Subset al (örn: 10000 train, 2000 test)
# x_train = x_train[:10000]
# y_train = y_train[:10000]

# x_test = x_test[:2000]
# y_test = y_test[:2000]

# # 3️⃣ Normalize
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # 4️⃣ Resize + channel ekle (28x28 → 224x224x3)
# def preprocess(images):
#     images = tf.image.resize(images[..., np.newaxis], (224, 224))
#     images = tf.image.grayscale_to_rgb(images)
#     return images

# x_train = preprocess(x_train)
# x_test = preprocess(x_test)

# # 5️⃣ AlexNet benzeri model
# model = models.Sequential([
#     layers.Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(224,224,3)),
#     layers.MaxPooling2D((3,3), strides=2),

#     layers.Conv2D(256, (5,5), padding='same', activation='relu'),
#     layers.MaxPooling2D((3,3), strides=2),

#     layers.Conv2D(384, (3,3), padding='same', activation='relu'),
#     layers.Conv2D(384, (3,3), padding='same', activation='relu'),
#     layers.Conv2D(256, (3,3), padding='same', activation='relu'),
#     layers.MaxPooling2D((3,3), strides=2),

#     layers.Flatten(),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(10, activation='softmax')
# ])

# # 6️⃣ Compile
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # 7️⃣ Train
# model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# # 8️⃣ Test
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("Test accuracy:", test_acc) #0.9679999947547913

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# sklearn & plot
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1️⃣ MNIST yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Subset al
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

# 3️⃣ Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4️⃣ Resize + channel
def preprocess(images):
    images = tf.image.resize(images[..., np.newaxis], (224, 224))
    images = tf.image.grayscale_to_rgb(images)
    return images.numpy()   # 🔥 önemli (tensor → numpy)

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# 5️⃣ Model (AlexNet benzeri - hafifletildi)
model = models.Sequential([
    layers.Input(shape=(224,224,3)),

    layers.Conv2D(96, (11,11), strides=4, activation='relu'),
    layers.MaxPooling2D((3,3), strides=2),

    layers.Conv2D(256, (5,5), padding='same', activation='relu'),
    layers.MaxPooling2D((3,3), strides=2),

    layers.Conv2D(384, (3,3), padding='same', activation='relu'),
    layers.Conv2D(384, (3,3), padding='same', activation='relu'),
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((3,3), strides=2),

    layers.Flatten(),

    # 🔥 hafiflettik (4096 çok ağır)
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

# 6️⃣ Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7️⃣ Train
model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 8️⃣ Test
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# 🔥 9️⃣ Tahmin (yeniden eğitim yok)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 🔥 10️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

# 🔥 11️⃣ Classification Report
report = classification_report(y_test, y_pred_classes, output_dict=True)
df = pd.DataFrame(report).transpose()

print(df)
df.to_csv("classification_report.csv")

# 🔥 12️⃣ Tahminleri kaydet
results = pd.DataFrame({
    "Gerçek": y_test,
    "Tahmin": y_pred_classes
})
results.to_csv("tahminler.csv", index=False)