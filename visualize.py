import tensorflow as tf
import numpy as np
from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from src.chroma_rolls_preprocessor import get_chroma_from_midi, get_midi_from_chroma
import numpy as np
import tensorflow as tf
import os
import argparse

def hw_interpolate(model,latent_size, steps):
    S = steps
    z0 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.reshape(z, (S * S, latent_size))  # [S, S, latent_size]
    x = model.decoder(z)

def get_latent_encoding(model, chroma):
    chroma = np.expand_dims(chroma, 0)
    latent_encoding = model.get_latent_encoding(chroma)
    return tf.squeeze(latent_encoding, 0)

def interpolate(model, file1, file2, latent_size, steps):
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
    chroma0 = get_chroma_from_midi(file1)
    chroma1 = get_chroma_from_midi(file2)
    z0 = get_latent_encoding(model,chroma0) # [S, latent_size]
    z1 = get_latent_encoding(model,chroma1)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.squeeze(z,0)
    x = model.decoder(z)  # [S]
    print(x.shape)

def predict_and_write_midi(model, midi_file):
    chroma = get_chroma_from_midi(midi_file)
    orig_midi = get_midi_from_chroma(chroma, tempo=120)
    orig_midi.write('original.mid')
    chroma_batch = np.expand_dims(chroma, 0)
    pred_chroma = model.predict(chroma_batch)
    pred_chroma = tf.squeeze(pred_chroma, axis=0).numpy()
    pred_midi = get_midi_from_chroma(pred_chroma, tempo=120)
    pred_midi.write('model-reconstruction.mid')

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

    # test_dir = 'data/test_data'
    # files = os.listdir(test_dir)
    # processed = []
    # for f in files:
    #     path = os.path.join(test_dir,f)
    #     test_midi_processed = get_chroma_from_midi(path)
    #     processed.append(test_midi_processed)
    # processed = np.array(processed)
    # model_midi_processed = model.predict(processed)

    test_midi_file = 'data/Dancing Queen.mid'
    test_midi_file2 = 'data/africa.mid'
    #predict_and_write_midi(model, test_midi_file)
    predict_and_write_midi(model, test_midi_file2)
    #interpolate(model,test_midi_file, test_midi_file2,32,3)
