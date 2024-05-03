import tensorflow as tf
import numpy as np
from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.variational_autoencoder import VAE
import numpy as np
import tensorflow as tf
import os
import argparse


def hw_interpolate(latent_size, steps):
    S = steps
    z0 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    print('w',w.shape)
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.reshape(z, (S * S, latent_size))  # [S, S, latent_size]
    print('z',z.shape)

def interpolate(model, latent_size, steps):
    """
    Call this only if the model is VAE!
    Generate interpolation between two .
    Show the generated images from your trained VAE.
    Image will be saved to outputs/show_vae_interpolation.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """
    S = steps
    z0 = tf.random.normal(shape=[latent_size,], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[latent_size,], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    print('w',w.shape)
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    print('z',z.shape)
    #x = model.decoder(z)  # [S]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=int, required = True, help = "0 for 128 pitches, 1 for 12")
    parser.add_argument("-model", type=str, required = True, help = "model file name", default= "vae-defaul.keras")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if (args.f == 0):
        processing = get_midi_from_data_eric
    elif (args.f == 1):
        processing = get_midi_from_data_evan
    
    model_path = "saved_model/" + args.model
    model = tf.keras.models.load_model(model_path)
    model.summary()
    test_midi_file = "data/data/lyricsMidisP0/" 
    test_midi_file = 'data/data/Dancing Queen.mid'
    test_midi_processed = processing(test_midi_file)
    model_midi_processed = model.predict(test_midi_processed)
    reconstructed_midi = get_midi_from_data(model_midi_processed)