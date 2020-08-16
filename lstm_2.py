import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

series = []
time = []
print(tf.keras.losses.mean_absolute_error([1.0,2.0,3.0],[2.0,2.0,2.0]).numpy())
with open("sunspot.csv") as file:
    csv_reader = csv.reader(file,delimiter=",")
    next(csv_reader)
    for row in csv_reader:
        time.append(int(row[0]))
        series.append(float(row[2]))

series = np.array(series)
time = np.array(time)

SPLIT_TIME = 1000

series_train = series[:SPLIT_TIME]
time_train = time[:SPLIT_TIME]
series_valid = series[SPLIT_TIME:]
time_valid = time[SPLIT_TIME:]

def make_windowed_dataset(series,window_size,batch_size,shuffle_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=window_size+1,shift=1,drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size+1))
    ds = ds.shuffle(shuffle_size).map(lambda window: (window[:-1], window[-1]))
    ds = ds.map(lambda x,y: (tf.expand_dims(x,axis=-1),y))
    return ds.batch(batch_size).prefetch(2)

def easy_print(k):
    for a in k:
        print(a)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

window_size = 25
batch_size = 32
shuffle = 2900
dataset = make_windowed_dataset(series_train,window_size,batch_size,shuffle)
OPTIM = tf.keras.optimizers.Adam()
LOSSES = tf.keras.losses.MeanSquaredError()

def get_model_1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32,return_sequences=True,input_shape=[window_size,1]),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(100,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.relu)
    ])
    model.compile(optimizer=OPTIM,loss='mse',metrics=['mse'])
    print(model.summary())
    return model

def predict_and_plot(model):
    forecast = []
    result = []
    for time in range(len(series) - window_size):
        item = np.expand_dims(series[time:time+window_size][np.newaxis],axis=-1)
        res = model.predict(item)
        forecast.append(res)

    forecast = forecast[SPLIT_TIME - window_size:]
    result = np.array(forecast)[: ,0, 0]
    mae = tf.keras.metrics.mean_absolute_error(series_valid,result).numpy()
    plot_series(time_valid,series_valid)
    plot_series(time_valid,result)
    plt.title("MAE: {}".format(mae))
    plt.show()



model_1 = get_model_1()
model_1.fit(dataset,epochs=100)

predict_and_plot(model_1)
print("WHERE")