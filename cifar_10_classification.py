import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

print(sys.version)
print(tf.__version__)
(ds_train,ds_test) = tfds.load("cifar10",as_supervised=True,split=['train','test'])
class_names = ['auto']

def show_sample_image(ds):
    i = 1
    for img,label in ds.take(4):
        ax = plt.subplot(2,2,i)
        ax.imshow(np.squeeze(img.numpy()))
        ax.set_title(label.numpy())
        plt.show()

def parse_images(img,label):
    return tf.cast(img,tf.float32),label

def augment_image(img,label):
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    return img,label
BATCH = 32
EPOCHS = 10
ds_train = ds_train.map(parse_images).map(augment_image).batch(BATCH).prefetch(2).shuffle(100)
ds_test = ds_test.map(parse_images).batch(BATCH)

optim = tf.keras.optimizers.RMSprop()
losses = tf.keras.losses.SparseCategoricalCrossentropy()

def get_model_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Dense(384,activation=tf.nn.swish),
        tf.keras.layers.Dense(256,activation=tf.nn.swish),
        tf.keras.layers.Dense(128,activation=tf.nn.swish),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optim, loss=losses, metrics=['accuracy'])
    return model

def get_model_2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,3,input_shape=(32,32,3),activation=tf.nn.relu),
        tf.keras.layers.Conv2D(32,3,activation=tf.nn.relu),
        tf.keras.layers.Conv2D(16,3,activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(384,activation=tf.nn.swish),
        tf.keras.layers.Dense(256,activation=tf.nn.swish),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optim,loss=losses,metrics=['accuracy'])
    return model

first_model = get_model_2()
hist = first_model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test)

def plot_accuracy_and_loss(hist):
    acc = hist.history['accuracy']
    acc_val = hist.history['val_accuracy']
    loss = hist.history['loss']
    loss_val = hist.history['val_loss']
    ax = plt.subplot(2,1,1)
    ax.plot(acc,label='acc')
    ax.plot(acc_val,label='val_acc')
    ax.legend()
    ax.set_title("Acc over Time")
    ax.set_xlabel("Epochs")

    ax = plt.subplot(2, 1, 2)
    ax.plot(loss,label='train_loss')
    ax.plot(loss_val,label='val_loss')
    ax.legend()
    ax.set_title("Loss over Time")
    ax.set_xlabel("Epochs")
    plt.show()
    plt.legend()
    plt.tight_layout()

plot_accuracy_and_loss(hist)