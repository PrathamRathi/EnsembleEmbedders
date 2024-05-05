import tensorflow as tf
import numpy as np
from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from src.chroma_rolls_preprocessor import get_chroma_from_midi, get_midi_from_chroma
import numpy as np
import tensorflow as tf
from net import variational_autoencoder
import os
import argparse

INFERENCE_DIR = 'inference/'
ORIGINAL = '-original.mid'
RECONSTRUCTED = '-model-reconstructed.mid'

def hw_interpolate(model,latent_size, steps):
    S = steps
    z0 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.reshape(z, (S * S, latent_size))  # [S, S, latent_size]
    x = model.decoder(z)

def get_latent_encoding(model, chroma)->tf.Tensor:
    """
    Gets latent encoding for the given chroma
    Inputs:
    - model: a trained tf.keras.Model instance
    - chroma: numpy array that is the chroma representation of the midi in choice
    Returns: a tensor that is the latent encoding of the chroma generated by the model
    """
    chroma = np.expand_dims(chroma, 0).astype(np.int32)
    latent_encoding = model(chroma)[-1] # returns z
    # return tf.squeeze(latent_encoding, 0)
    return latent_encoding

def interpolate(model, file0, file1, steps, name):
    """
    Generate interpolation between two midi files
    Inputs:
    - model: a trained tf.keras.Model instance
    - file0, file1: paths to midi files that will be the start and end of the interpolation respectively
    - latent_size: Latent size of your model.
    - steps: number of steps between (and inclusive of) the start and end
    Returns: a tensor that is of the same shape as model output with steps number as the batch size
    """
    chroma0 = get_chroma_from_midi(file0)
    chroma1 = get_chroma_from_midi(file1)
    z0 = get_latent_encoding(model,chroma0)
    z1 = get_latent_encoding(model,chroma1)
    w = tf.linspace(0, 1, steps)
    w = tf.cast(tf.reshape(w, (steps, 1, 1)), dtype=tf.float32)  
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.squeeze(z,0)
    x = model.decoder(z)
    for i in range(steps):
        chroma = x[i]
        chroma = chroma.numpy()
        chroma_to_file(chroma, INFERENCE_DIR + str(i) + name)

def chroma_to_file(chroma, file_path):
    """
    Converts chroma matrix to a midi and writes it to a file
    Inputs:
    - chroma: a numpy matrix that represents chroma representation of a midi
    - file_path: path to write midi to
    """
    midi = get_midi_from_chroma(chroma, tempo=120)
    midi.write(file_path)

def predict_and_write_midi(model, midi_file, name):
    """
    Gets model inference given a midi file and writes original midi and reconstructed midi
    Inputs:
    - model: a trained tf.keras.Model instance
    - midi_file: a string that is the path to the midi file of choice
    - name: a string that will be prepended to the written midi files
    """
    chroma = get_chroma_from_midi(midi_file)
    chroma_to_file(chroma, INFERENCE_DIR + name + ORIGINAL)
    chroma_batch = np.expand_dims(chroma, 0).astype(np.int32)
    pred_chroma = model(chroma_batch)[0]
    pred_chroma = tf.squeeze(pred_chroma, axis=0).numpy()
    chroma_to_file(pred_chroma, INFERENCE_DIR + name + RECONSTRUCTED)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help = "pitches or chroma", default="chroma")
    parser.add_argument("-model", type=str, help = "model file name", default= "vae-default")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # if (args.f == "pitches"):
    #     processing = get_data_from_midi
    # elif (args.f == "chroma"):
    #     processing = get_chroma_from_midi
    
    model_path = "saved_model/" + args.model
    model = tf.keras.models.load_model(model_path)
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

    test_midi_file0 = 'data/Dancing Queen.mid'
    test_midi_file1 = 'data/africa.mid'
    # predict_and_write_midi(model, test_midi_file, 'dq')
    #predict_and_write_midi(model, test_midi_file2, 'toto')
    interpolate(model,test_midi_file0, test_midi_file1,3,'dq-toto.mid')
