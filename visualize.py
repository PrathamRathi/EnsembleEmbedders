import tensorflow as tf
import numpy as np
from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from src.chroma_rolls_preprocessor import get_chroma_from_midi, get_midi_from_chroma
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
    parser.add_argument("-f", type=str, help = "pitches or chroma", default="chroma")
    parser.add_argument("-model", type=str, help = "model file name", default= "default.keras")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # if (args.f == "pitches"):
    #     processing = get_data_from_midi
    # elif (args.f == "chroma"):
    #     processing = get_chroma_from_midi
    
    model_path = "saved_model/" + args.model
    model = tf.keras.models.load_model(model_path)
    model.build(input_shape = (1,3, 12, 160))
    model.summary()
    test_midi_file = "data/data/lyricsMidisP0/" 

    # test_dir = 'data/test_data'
    # files = os.listdir(test_dir)
    # processed = []
    # for f in files:
    #     path = os.path.join(test_dir,f)
    #     test_midi_processed = get_chroma_from_midi(path)
    #     processed.append(test_midi_processed)
    # processed = np.array(processed)
    # print(processed.shape)
    # model_midi_processed = model.predict(processed)

    test_midi_file = 'data/Dancing Queen.mid'
    test_midi_processed = get_chroma_from_midi(test_midi_file)
    test_midi_processed = np.expand_dims(test_midi_processed, 0)
    model_midi_processed = model.predict(test_midi_processed)
    model_midi_processed = tf.squeeze(model_midi_processed, axis=0).numpy()
    print(model_midi_processed)
    reconstructed_midi = get_midi_from_chroma(model_midi_processed, tempo=120)
    reconstructed_midi.write('model-reconstruction.mid')