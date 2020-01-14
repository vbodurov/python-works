# create network with layers:
# layer 1: 28x28=274 neurons
# layer 2: 128 neurons
# layer 3: 10 neurons one for each label
# compile it with: model.compile...
# train it with: model.fit...
# evaluate how well it is doing with: model.evaluate...

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

start = time.time()
test_loss, test_acc = model.evaluate(test_images, test_labels)
end = time.time()
print("Evaluation Seconds:", end - start)
print("Tested Acc:", test_acc)
