import tensorflow as tf

class Autoencoder(tf.keras.Model):
    def __init__(self, epochs, instrument_units, pitch_units, song_length, learning_rate):
        super(Autoencoder, self).__init__()

        self.epochs = epochs
        self.instrument_units = instrument_units
        self.pitch_units = pitch_units
        self.song_length = song_length

        self.loss = tf.keras.losses.MeanSquaredError
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        input_shape = (self.instrument_units, self.pitch_units, self.song_length)
        flattened_dim = self.pitch_units * self.song_length * self.instrument_units
        if (instrument_units == 1):
            input_shape = (self.pitch_units, self.song_length)
            flattened_dim = self.pitch_units * self.song_length

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(),
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(flattened_dim),
            tf.keras.layers.Reshape(input_shape)
        ])

    def call(self, x, verbose = False):
        if (verbose):
            for i, layer in enumerate(self.encoder.layers):
                x = layer(x)
                print("encoder layer: ", layer, "output dim", x.shape)
            for i, layer in enumerate(self.decoder.layers):
                x = layer(x)
                print("decoder layer: ", layer, "output dim", x.shape)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
        return x

class LossAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            print("\nEpoch {} - Loss: {:.4f}".format(epoch, logs['loss']))
            
if __name__ == "__main__":
    print("executing autoencoder.py..")
    # data = get_data()
    # autoencoder = Autoencoder()
    # autoencoder.compile(
    #     optimizer = autoencoder.optimizer,
    #     loss = autoencoder.loss
    # )
    # autoencoder.fit(data, data, 
    #                       epochs=autoencoder.epochs, 
    #                       batch_size=32, 
    #                       validation_split=0.2,
    #                       callbacks=[LossAccuracyCallback()])
    # loss = autoencoder.evaluate(data, data)
    # reconstructed_songs = autoencoder.predict(data)
    
