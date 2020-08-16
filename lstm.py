import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def easy_print(k):
    for a in k:
        print(list(a.as_numpy_iterator()))

def make_window_dataset(series,window_size,batch_size,buffer_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size,shift=1,drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    ds = ds.shuffle(buffer_size).map(lambda w: (w[:-1],w[-1]))
    return ds.batch(batch_size).prefetch(2)

train = make_window_dataset(series,window_size=20,batch_size=32,buffer_size=100)

LSTM_CELL = 64
OPTIM = tf.keras.optimizers.SGD()
LOSSES = tf.keras.losses.Huber()
LR_SCHEDULER = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]),
        tf.keras.layers.LSTM(LSTM_CELL,return_sequences=True),
        tf.keras.layers.LSTM(LSTM_CELL),
        tf.keras.layers.Dense(10,activation=tf.nn.relu),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100)
    ])
    model.compile(loss=LOSSES,optimizer=OPTIM,metrics=['mae'])
    print(model.summary())
    return model

model = get_model()
model.fit(train,epochs=20,callbacks=[LR_SCHEDULER])

forecast = []
results = []
for time in range(len(series) - window_size):
    res = model.predict(series[time:time+window_size][np.newaxis])
    forecast.append(res)

forecast = forecast[split_time - window_size:]
result = np.array(forecast)[:,0, 0]

plot_series(time_valid, x_valid)
plot_series(time_valid, result)
plt.show()

