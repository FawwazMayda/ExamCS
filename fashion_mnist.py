import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(ds_train,ds_test) = tfds.load("fashion_mnist",split=['train','test'],as_supervised=True,download=False)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# See the images
i = 1

# Numpy expect Images in (28,28)
# TensorFlow expect in (28,28,1)

def parse_images(img,label):
    img = tf.cast(img,tf.float32) / 255.0
    return img,label

def data_augment(img,label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img,label

BATCH = 64
EPOCHS = 1000
ds_train = ds_train.map(parse_images).batch(BATCH).prefetch(2).shuffle(100)
ds_test = ds_test.map(parse_images).batch(BATCH)

optim = tf.keras.optimizers.SGD()
losses = tf.keras.losses.SparseCategoricalCrossentropy()

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(245,activation=tf.nn.relu),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optim,loss=losses,metrics=['accuracy'])
    return model

fmnist_model = get_model()

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\n Starting training; got log keys: {}".format(keys))
        print("\n got no logs")

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\n Starting training; got log keys: {}".format(keys))
        acc = logs['accuracy']
        if acc > 0.85:
            self.model.stop_training=True
            print("Model Reach above 85% Accuracy")

hist = fmnist_model.fit(ds_train,epochs=EPOCHS,callbacks=[CustomCallbacks()])

def plot_accuracy_and_loss(hist):
    acc = hist.history['accuracy']
    loss = hist.history['loss']
    ax = plt.subplot(2,1,1)
    ax.plot(acc)
    ax.set_title("Acc vs Epochs")
    ax.set_xlabel("Epochs")

    ax = plt.subplot(2, 1, 2)
    ax.plot(loss)
    ax.set_title("Acc vs Loss")
    ax.set_xlabel("Epochs")
    plt.show()
    plt.tight_layout()

plot_accuracy_and_loss(hist)