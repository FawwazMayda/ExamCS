import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
text = ["CountryBall and CountryBox","SwissBall is CountryBall","SwissBox is CountryBox"]

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<B>")
tokenizer.fit_on_texts(text)

print(tokenizer.word_index)
test_text = [
    'CountryBall CountryBox',"SwissBall is CountryBall Not CountryBox",
    "abang bola"
]
seq = tokenizer.texts_to_sequences(test_text)
seq_pad = pad_sequences(seq,maxlen=4,padding='post')
print(seq_pad)