from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.variational_autoencoder import VAE
import numpy as np
import tensorflow as tf
import os
import argparse

#-epochs 10 -lr 3e-5 -file chroma_rolls_all.npy
# data_path = 'data/data/lyricsMidisP0'
data_path = 'data/data/lyricsMidisP0'
output_path = 'output/'
preprocessed_folder_path = "preprocessed/"
default_preprocessed_path = 'preprocessed/all_chunks' # all_chunks is the filename of the .npy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, required = True, help = "epochs")
    parser.add_argument("-lr", type=float, required = True, help = "learning rate")
    parser.add_argument("-n", type=int, help = "number of midis if processing needed")
    parser.add_argument("-file", type=str, required = True, help = "preprocessed .npy file path", default= "default")
    return parser.parse_args()

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
            if (verbose):
                print('Processing file(s) {}.'.format(i))
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
        
    # Create the output directory if it doesn’t exist yet
    if not os.path.exists(os.path.dirname(preprocessed_path)):
        os.makedirs(os.path.dirname(preprocessed_path))
        
    np.save(preprocessed_path, all_chunks)
    print("Saved preprocessing output to {}".format(preprocessed_path))


if __name__ == "__main__":
    # Preprocess data and write output file as .npy
    #   Comment this line if using cached preprocessed files!
    args = parse_arguments()
    if (args.file != "default"):
        preprocessed_path = preprocessed_folder_path + args.file
    else:
        preprocessed_path = default_preprocessed_path + "_" + str(args.n)

    #if file already exists
    if (not os.path.exists(preprocessed_path)):
        preprocess(data_path, preprocessed_path, num_files=args.n)

    # Load data from .npy and train
    #TODO: automatically load only the batch size into data
    data = np.load(preprocessed_path)
    x_train = tf.constant(data)
    print(f"Loading preprocessed data from file {preprocessed_path}. Shape: {x_train.shape}")
    
    #TODO: automatically get the parameters below
    instrument_units = 3
    pitch_units = 12
    song_length = 160
    model = VAE(song_length= song_length,
                        instrument_units= instrument_units,
                        pitch_units= pitch_units,
                        learning_rate= args.lr,
                        epochs=args.epochs,
                        hidden_dim=512,latent_size=32
                        )
    model.compile(
        optimizer = model.optimizer,
    )
    model.build(input_shape = (1,3, 12, 160))
    model.summary()
    model.fit(x_train, 
              x_train,
                epochs=model.epochs,
              batch_size = 32 if model.epochs > 300 else 10,
              validation_split = 0.2,
            #   callbacks = [LossAccuracyCallback()]
    )
    print('Saving model')
    model.save("saved_model/vae-default.keras")