import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix

# 🔥 1) DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# subset (hız için)
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:200]
y_test = y_test[:200]

# normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# resize + RGB
def preprocess(images):
    images = tf.image.resize(images[..., np.newaxis], (224, 224))
    images = tf.image.grayscale_to_rgb(images)
    return images

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# 🔥 MODEL FONKSİYONU
def run_model(model_name):

    if model_name == "ResNet50":
        base = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224,224,3))

    elif model_name == "ResNet101":
        base = tf.keras.applications.ResNet101(weights=None, include_top=False, input_shape=(224,224,3))

    elif model_name == "EfficientNet":
        base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))

    # classifier
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
                   
    print(f"\n🚀 {model_name} eğitim başlıyor...\n")

    model.fit(
        x_train, y_train,
        epochs=16,
        batch_size=8,
        validation_data=(x_test, y_test),
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test)
    print(f"{model_name} Accuracy:", acc)

    # 🔥 Tahmin
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 🔥 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

    plt.savefig(f"{model_name}_cm.png")
    plt.close()

    # 🔥 sonuç kaydet
    df = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": acc
    }])

    df.to_csv("results.csv", mode='a', index=False,
              header=not os.path.exists("results.csv"))

    return acc

# 🔥 2) MODELLERİ ÇALIŞTIR
models = ["ResNet50", "ResNet101", "EfficientNet"]

for m in models:
    run_model(m)

# 🔥 3) TEK GRAFİK
df = pd.read_csv("results.csv")

plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["Accuracy"], color=["blue","green","orange"])
plt.title("Model Karşılaştırma")
plt.ylabel("Accuracy")

plt.savefig("final_comparison.png")
plt.show()

# 🔥 4) CONFUSION MATRİSLERİ TEK GÖRSEL
fig, axes = plt.subplots(1,3, figsize=(15,4))

for i, name in enumerate(models):
    img = plt.imread(f"{name}_cm.png")
    axes[i].imshow(img)
    axes[i].set_title(name)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("all_confusion_matrices.png")
plt.show()

print("\n✅ TÜM MODELLER TAMAMLANDI")