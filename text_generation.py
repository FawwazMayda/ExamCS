import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

corpus = ['CountryBall is a Ball',
          'SwissBall is a CountryBall',
          'Abang Bola Ball Ball Ball',
          'Bola Oh Bola Mengapa Engkau Bola',
          'Bola Oh Bola',
          'Oh Bola Oh',
          'Abang Bola Ball',
          'CountryBall Ball Ball','Hey Ball','Ball is a Ball','Hello Ball']

token = Tokenizer()
token.fit_on_texts(corpus)
print(token.word_index)
print(token.index_word)
print(token.word_docs)
TOTAL_WORDS = len(token.word_index) + 1
n_gram_seq = []
for line in corpus:
    seq = token.texts_to_sequences([line])[0]
    for i in range(len(seq)):
        temp_seq = seq[:i+1]
        n_gram_seq.append(temp_seq)

print("SEQUENCES")
for l in n_gram_seq:
    print(l)

print("PADDED SEQUENCES")
MAX_PAD_SEQ = max([len(k) for k in n_gram_seq])
padded_sequences = pad_sequences(n_gram_seq,maxlen=MAX_PAD_SEQ,padding='pre')
for l in padded_sequences:
    print(l)

print("FEATURE AND LABEL EXAMPLE")
xs,ys =[],[]
for l in padded_sequences:
    xs.append(l[:-1])
    ys.append(l[-1])
ys_oh = tf.keras.utils.to_categorical(ys)
xs = np.array(xs)
ys= np.array(ys)
ys_oh = np.array(ys_oh)

for i in range(7):
    print("Sentences: {}".format(padded_sequences[i]))
    print("Actual Sentences: {}".format(token.sequences_to_texts([padded_sequences[i]])[0]))
    print("Feature: {}".format(xs[i]))
    print("Label: {}".format(ys[i]))

print("DATA SHAPE")
print(xs.shape)
print(ys.shape)
print(ys_oh.shape)

def seq_and_pad(s,return_seq=False):
    seq = token.texts_to_sequences([s])
    padded = pad_sequences(seq,maxlen=MAX_PAD_SEQ,padding='pre')
    if return_seq:
        return padded
    else:
        return padded[0]
print(seq_and_pad("Ball CountryBall"))


LOSSES = tf.keras.losses.CategoricalCrossentropy()
OPTIM = tf.keras.optimizers.Adam()
EMBED_DIM = 8

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(TOTAL_WORDS,EMBED_DIM,input_length=MAX_PAD_SEQ),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,activation=tf.nn.relu),
        tf.keras.layers.Dense(16,activation=tf.nn.relu),
        tf.keras.layers.Dense(TOTAL_WORDS,activation=tf.nn.softmax)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['accuracy'])
    return model

model = get_model()
print(model.summary())
model.fit(xs,ys_oh,epochs=100,batch_size=1)

def parse_pred(model,s):
    seq = seq_and_pad(s,return_seq=True)
    pred = model.predict(seq)
    print("Actual Pred: {}".format(pred))
    pred_idx = np.argmax(pred,axis=1)
    print("Argmax: {}".format(pred_idx))
    pred_word = token.index_word[int(pred_idx[0])]
    print("Sentences: {}".format(s))
    print("Predicted next Word: {}".format(pred_word))
    print("\n")

parse_pred(model,"Abang Abang Abang")
parse_pred(model,"Bola")
parse_pred(model,"oh")
parse_pred(model,"CountryBall")