import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DIR = "Chessman-image-dataset/Chess"
BATCH = 4
EPOCHS = 10
IMG_HEIGT = 64
IMG_WIDTH = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                          width_shift_range=0.2,height_shift_range=0.2,
                                                          zoom_range=0.2,shear_range=0.2,horizontal_flip=True,
                                                          validation_split=0.25)

train_gen = datagen.flow_from_directory(directory=TRAIN_DIR,batch_size=BATCH,
                                        target_size=(IMG_WIDTH,IMG_HEIGT),subset='training'
                                        ,class_mode='categorical')
test_gen = datagen.flow_from_directory(directory=TRAIN_DIR,batch_size=BATCH,
                                       target_size=(IMG_WIDTH,IMG_HEIGT),subset='validation',
                                       class_mode='categorical')

optim = tf.keras.optimizers.Adam()
losses = tf.keras.losses.CategoricalCrossentropy()

def get_model_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,3,input_shape=(IMG_WIDTH,IMG_HEIGT,3),activation=tf.nn.relu),
        tf.keras.layers.Conv2D(32,3,activation=tf.nn.relu),
        tf.keras.layers.Conv2D(16,3,activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.swish),
        tf.keras.layers.Dense(32,activation=tf.nn.swish),
        tf.keras.layers.Dense(6,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optim,loss=losses,metrics=['accuracy'])
    return model

def get_model_2():
    base = tf.keras.applications.mobilenet.MobileNet(input_shape=(IMG_WIDTH,IMG_HEIGT,3),include_top=False,pooling='max')
    base.trainable = False
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.swish),
        tf.keras.layers.Dense(32,activation=tf.nn.swish),
        tf.keras.layers.Dense(6,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optim,loss=losses,metrics=['accuracy'])
    return model

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

base_model = get_model_2()
hist1 = base_model.fit(train_gen,epochs=EPOCHS,validation_data=test_gen)
plot_accuracy_and_loss(hist1)