import tensorflow as tf
from src.midi_utils import get_data

class Autoencoder(tf.keras.Model):
    def __init__(self, epochs = 100, instrument_units = 5, timestamp_fq = 1 / 100, pitch_units = 128):
        super(Autoencoder, self).__init__()
        self.epochs = epochs
        self.instrument_units = instrument_units
        self.timestamp_frequency = timestamp_fq
        self.pitch_units = pitch_units
        self.song_length = (1 / self.timestamp_frequency ) * 60 * 60 * 5

        self.loss = tf.keras.losses.MeanSquaredError
        self.optimizer = tf.keras.optimizers.Adam
        input_shape = (self.instrument_units, self.pitch_units, self.song_length)
        self.encoder_layers = [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(1028, activation='leakyRelu'),
            tf.keras.layers.Dense(512, activation='leakyRelu'),
            tf.keras.layers.Dense(256, activation='leakyRelu'),
            tf.keras.layers.Dense(128, activation='leakyRelu')
        ]
        
        self.decoder_layers = [
            tf.keras.layers.Dense(128, activation='leakyRelu'),
            tf.keras.layers.Dense(256, activation='leakyRelu'),
            tf.keras.layers.Dense(512, activation='leakyRelu'),
            tf.keras.layers.Dense(1028, activation='leakyRelu'),
            tf.keras.layers.Reshape(input_shape)
        ]

    def call(self, x):
        x = x
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x

class LossAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print("\nEpoch {} - Loss: {:.4f}".format(epoch, logs['loss']))
if __name__ == "__main__":
    data = get_data()
    autoencoder = Autoencoder()
    autoencoder.compile(
        optimizer = autoencoder.optimizer,
        loss = autoencoder.loss
    )
    autoencoder.fit(data, data, 
                          epochs=autoencoder.epochs, 
                          batch_size=32, 
                          validation_split=0.2,
                          callbacks=[LossAccuracyCallback()])
    loss = autoencoder.evaluate(data, data)
    reconstructed_songs = autoencoder.predict(data)
