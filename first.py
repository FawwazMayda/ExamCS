import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

print("TF Version: {}".format(tf.__version__))

ds_train = tfds.load("iris",as_supervised=True,split='train',download=True)
ds_test = tfds.load("iris",as_supervised=True,split='train',download=True)
print(type(ds_train))
# Inspect
for feature,labels in ds_train.take(4):
    print(feature)
    print(labels)

# Load as Numpy and use Batch
"""
ds_numpy = tfds.load("iris",as_supervised=True,split='train')
for feature,labels in tfds.as_numpy(ds_numpy.take(1)):
    print(feature.shape)
    print(labels.shape)

ds_numpy = tfds.as_numpy(tfds.load("iris",as_supervised=True,split='train',batch_size=-1))
(feat,lbl) = ds_numpy
print(feat.shape)
print(lbl.shape)
"""

def parse_feat_and_label(feat,label):
    return feat,label

ds_train = ds_train.map(parse_feat_and_label).batch(32).prefetch(2).shuffle(100)
ds_test = ds_test.map(parse_feat_and_label).batch(32)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(4)),
    tf.keras.layers.Dense(100,activation=tf.nn.relu),
    tf.keras.layers.Dense(3,activation=tf.nn.softmax)
])
sgd = tf.keras.optimizers.SGD()
losses = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=sgd,loss=losses,metrics=['accuracy'])

model.fit(ds_train,epochs=20)
pred = model.predict(ds_test)
print("Raw Output:")
print(pred[:10])
print("Argmax Output:")
print(np.argmax(pred[:10],axis=1))