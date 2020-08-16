import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

imdb,info = tfds.load("imdb_reviews",as_supervised=True,with_info=True)
print(imdb.keys())

train_data,test_data = imdb['train'],imdb['test']
train_sentences,train_labels =[],[]
test_sentences,test_labels = [],[]

for s,l in train_data.take(2):
    print(str(s.numpy()))
    print(l.numpy())

for s,l in train_data:
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s,l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

VOCAB_SIZE = 3000
EMBEDDING_DIM = 128
MAX_PAD_LENGTH = 100
PADDING_OPTION = 'post'
TRUNC_OPTION = 'post'

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE,oov_token="<B>")
tokenizer.fit_on_texts(train_sentences)
train_sequence = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(train_sentences)
                                                               ,maxlen=MAX_PAD_LENGTH,
                                                               truncating=TRUNC_OPTION,padding=PADDING_OPTION)
test_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_sentences),
                                                               maxlen=MAX_PAD_LENGTH,
                                                               truncating=TRUNC_OPTION,padding=PADDING_OPTION)

print(train_sequence.shape)
print(train_labels.shape)
print(test_sequences.shape)
print(test_labels.shape)
print(train_labels[0])

EPOCHS = 10
OPTIM = tf.keras.optimizers.RMSprop()
LOSSES = tf.keras.losses.BinaryCrossentropy()

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

def get_model_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(MAX_PAD_LENGTH)),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

def get_model_2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_PAD_LENGTH),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

def get_model_3():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_PAD_LENGTH),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

def get_model_4():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_PAD_LENGTH),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

def get_model_5():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_PAD_LENGTH),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

def get_model_6():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,EMBEDDING_DIM,input_length=MAX_PAD_LENGTH),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model


model_1 = get_model_6()
print(model_1.summary())
hist1 = model_1.fit(train_sequence,train_labels,epochs=EPOCHS,batch_size=32,validation_data=(test_sequences,test_labels))
plot_accuracy_and_loss(hist1)