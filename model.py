import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


tf_model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(14, )),

    layers.Dense(16, activation='relu'),

    layers.Dense(1)
])
