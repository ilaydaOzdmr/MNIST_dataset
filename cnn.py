import tensorflow as tf
from tensorflow.keras import datasets, layers, models

print("1- Program başladı")

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print("2- Veri yüklendi:", x_train.shape, y_train.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
print("3- Veri hazırlandı")

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

print("4- Model kuruldu")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("5- Model derlendi")
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("6- Eğitim bitti")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

print("7- Program bitti")