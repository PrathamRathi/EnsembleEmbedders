import tensorflow as tf
import numpy as np
from src.chroma_rolls_preprocessor import get_chroma_from_midi, get_midi_from_chroma
import numpy as np
import tensorflow as tf
import pretty_midi as pm
import librosa.display
import matplotlib.pyplot as plt
import argparse
import json
import pprint

INFERENCE_DIR = 'inference/'
ORIGINAL = '-original.mid'
RECONSTRUCTED = '-model-reconstructed.mid'

def get_latent_encoding(model, chroma)->tf.Tensor:
    """
    Gets latent encoding for the given chroma
    Inputs:
    - model: a trained tf.keras.Model instance
    - chroma: numpy array that is the chroma representation of the midi in choice
    Returns: a tensor that is the latent encoding of the chroma generated by the model
    """
    chroma = np.expand_dims(chroma, 0).astype(np.int32)
    chroma = tf.transpose(chroma, perm=[0, 2, 3, 1])
    latent_encoding = model(chroma)[-1] # returns z
    # return tf.squeeze(latent_encoding, 0)
    return latent_encoding

def interpolate_by_average(model, file0, file1, weight):
    """
    Generate interpolation between two midi files by taking a weighted average of latent encodings
    Inputs:
    - model: a trained tf.keras.Model instance
    - file0, file1: paths to midi files that will be the start and end of the interpolation respectively
    - latent_size: Latent size of your model.
    - steps: number of steps between (and inclusive of) the start and end
    - name: base name of the midi file
    Returns: a tensor that is of the same shape as model output with steps number as the batch size
    """
    name0 = file0.split('/')[-1].split('.')[0]
    name1 = file1.split('/')[-1].split('.')[0]
    name = name0 + name1 + '.mid'
    chroma0 = get_chroma_from_midi(file0)
    chroma1 = get_chroma_from_midi(file1)
    z0 = get_latent_encoding(model,chroma0)
    z1 = get_latent_encoding(model,chroma1)
    z = weight * z0 + (1 - weight) * z1
    x = model.decoder(z)
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x = tf.squeeze(x).numpy()
    chroma_to_file(x, INFERENCE_DIR + 'average-' + name)
    return x

def interpolate_by_steps(model, file0, file1, steps):
    """
    Generate interpolation between two midi files by going step by step
    Inputs:
    - model: a trained tf.keras.Model instance
    - file0, file1: paths to midi files that will be the start and end of the interpolation respectively
    - steps: number of steps between (and inclusive of) the start and end
    - name: base name of the midi file
    Returns: a tensor that is of the same shape as model output with steps number as the batch size
    """
    name0 = file0.split('/')[-1].split('.')[0]
    name1 = file1.split('/')[-1].split('.')[0]
    name = name0 + name1 + '.mid'
    chroma0 = get_chroma_from_midi(file0)
    chroma1 = get_chroma_from_midi(file1)
    z0 = get_latent_encoding(model,chroma0)
    z1 = get_latent_encoding(model,chroma1)
    w = tf.linspace(0, 1, steps)
    w = tf.cast(tf.reshape(w, (steps, 1, 1)), dtype=tf.float32)
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.squeeze(z,0)
    x = model.decoder(z)
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    for i in range(steps):
        chroma = x[i]
        chroma = chroma.numpy()
        chroma_to_file(chroma, INFERENCE_DIR + 'step' + str(i) + name)
    return x

def chroma_to_file(chroma, file_path):
    """
    Converts chroma matrix to a midi and writes it to a file
    Inputs:
    - chroma: a numpy matrix that represents chroma representation of a midi
    - file_path: path to write midi to
    """
    midi = get_midi_from_chroma(chroma, tempo=120)
    plt.figure(figsize=(8, 4))
    plot_piano_roll(midi, 42, 90) # notes should be in 48 to 84
    # plt.show()
    basename = file_path.split('/')[-1]
    basename = basename.split('.')[0]
    plt.savefig(fname=INFERENCE_DIR + basename)
    midi.write(file_path)

def predict_and_write_midi(model, midi_file):
    """
    Gets model inference given a midi file and writes original midi and reconstructed midi
    Inputs:
    - model: a trained tf.keras.Model instance
    - midi_file: a string that is the path to the midi file of choice
    - name: a string that will be prepended to the written midi files
    """
    name = midi_file.split('/')[-1].split('.')[0]
    chroma = get_chroma_from_midi(midi_file)
    chroma_to_file(chroma, INFERENCE_DIR + name + ORIGINAL)
    chroma_batch = np.expand_dims(chroma, 0).astype(np.int32)
    chroma_batch = tf.transpose(chroma_batch, perm=[0, 2, 3, 1])
    pred_chroma = model(chroma_batch)[0]
    pred_chroma = tf.transpose(pred_chroma, perm=[0, 3, 1, 2])
    pred_chroma = tf.squeeze(pred_chroma, axis=0).numpy()
    chroma_to_file(pred_chroma, INFERENCE_DIR + name + RECONSTRUCTED)

# From https://github.com/craffel/pretty-midi/blob/main/Tutorial.ipynb
def plot_piano_roll(midi, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(midi.get_piano_roll(fs)[start_pitch:end_pitch],
                                hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                fmin=pm.note_number_to_hz(start_pitch))

def get_losses_from_history(model_path):
    model_name = model_path.split('/')[-1]
    with open('saved_model/history/' + model_name + '.json') as json_data:
        d = json.load(json_data)
        json_data.close()
    best_loss = d['loss'][-1]
    best_reconstruction_loss = d['recon. loss'][-1]
    best_KL_loss = d['kl loss'][-1]
    return best_loss, best_reconstruction_loss, best_KL_loss

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
    losses = get_losses_from_history(model_path)
    model.summary()
    print("Best losses from training:")
    print("loss: {}".format(losses[0]))
    print("recon. loss: {}".format(losses[1]))
    print("KL loss: {}".format(losses[2]))
    print()

    test_midi_file0 = 'data/dancing_queen.mid'
    test_midi_file1 = 'data/africa.mid'
    test_midi_file2 = 'data/wake_me_up.mid'
    test_midi_file3 = 'data/fly_me_to_the_moon.mid'

    # predict_and_write_midi(model, test_midi_file0, 'dq')
    # predict_and_write_midi(model, test_midi_file1, 'toto')
    predict_and_write_midi(model, test_midi_file0)
    predict_and_write_midi(model, test_midi_file1)

    interpolate_by_average(model,test_midi_file0, test_midi_file1, .5)
    interpolate_by_steps(model,test_midi_file0, test_midi_file1, 5)