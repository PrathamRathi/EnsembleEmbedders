from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.autoencoder import Autoencoder, LossAccuracyCallback
import numpy as np
import tensorflow as tf
import os

# data_path = 'data/data/lyricsMidisP0'
data_path = 'data/lyricsMidisP0'
output_path = 'output/'
preprocessed_path = 'preprocessed/all_chunks' # all_chunks is the filename of the .npy

def preprocess(data_path, preprocessed_path, num_files, verbose=False):
    '''
    Preprocesses midi files stored in data_path and writes out
    to an output file at the given path.

    Output is a .npy with shape=
        [num_samples, 128 pitches, 256 columns (4 bars of a song)]
    '''
    midis = get_midi_paths(data_path, depth=2)
    midis = midis[0][:num_files] # data is split into 5 sections, pick the first one.
    print('Preprocessing {} total files.'.format(len(midis)))

    all_chunks = None

    for i in range(len(midis)):
        try:
            # Get data from midi
            print('Processing file {}.'.format(i))
            melody_array = get_data_from_midi(midis[i], verbose=verbose)

            # Make all velocities 0 or 1
            melody_array = np.clip(melody_array, 0, 1)
            melody_array = melody_array.astype(np.ubyte)

            # Split into 4 bar segments
            COLUMNS_PER_BAR = 16 * 4
            BARS_PER_CHUNK = 4
            chunks = []
            col_step = COLUMNS_PER_BAR * BARS_PER_CHUNK
            for col_start in range(0, melody_array.shape[1], col_step):
                chunks.append(melody_array[:, col_start:col_start + col_step])
            chunks = chunks[:-1]

            if all_chunks is None:
                all_chunks = chunks
            else:
                all_chunks = np.concatenate([all_chunks, chunks], axis=0)

        except Exception as e:
            print(e)
            print('Continuing...')
            continue
        
    print("Shape of all_chunks: {}".format(all_chunks.shape))
        
    # Create the output directory if it doesn't exist yet
    if not os.path.exists(os.path.dirname(preprocessed_path)):
        os.makedirs(os.path.dirname(preprocessed_path))

    # Write out all chunks to file
    np.save(preprocessed_path, all_chunks)
    print("Saved preprocessing output to {}".format(preprocessed_path))


if __name__ == "__main__":
    # Preprocess data and write output file as .npy
    #   Comment this line if using cached preprocessed files!
    preprocess(data_path, preprocessed_path, num_files=50)

    # Load data from .npy and train
    data = np.load(preprocessed_path + ".npy")
    x_train = tf.constant(data)
    print("Loading preprocessed data from file. Shape: {}".format(x_train.shape))

    model = Autoencoder(song_length=256, # 256 columns = 4 bars
                        instrument_units= 1,
                        pitch_units=128
    )
    model.compile(
        optimizer = model.optimizer,
        loss = model.loss
    )
    model.fit(x_train, 
              x_train,
              epochs = model.epochs,
              batch_size = 2,
            #   validation_split = 0.2,
            #   callbacks = [LossAccuracyCallback()]
    )