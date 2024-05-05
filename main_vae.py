from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.variational_autoencoder import VAE
import numpy as np
import tensorflow as tf
import os
import argparse

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


if __name__ == "__main__":
    # Preprocess data and write output file as .npy
    #   Comment this line if using cached preprocessed files!
    args = parse_arguments()
    if (args.file != "default"):
        preprocessed_path = preprocessed_folder_path + args.file
    else:
        preprocessed_path = default_preprocessed_path + "_" + str(args.n)


    # Load data from .npy and train
    #TODO: automatically load only the batch size into data
    data = np.load(preprocessed_path)
    x_train = tf.constant(data)
    print(f"Loading preprocessed data from file {preprocessed_path}. Shape: {x_train.shape}")
    
    #TODO: automatically get the parameters below
    instrument_units = 3
    pitch_units = 12
    song_length = 160
    model = VAE(song_length= song_length,instrument_units= instrument_units,pitch_units= pitch_units,
                learning_rate= args.lr,epochs=args.epochs,
                        hidden_dim=512,latent_size=32)
    model.compile(optimizer = model.optimizer,)
    model.build(input_shape = (1,instrument_units, pitch_units, song_length))
    model.summary()
    model.fit(x_train, x_train,epochs=model.epochs,batch_size = 32 if model.epochs > 300 else 10,validation_split = 0.2)
    print('Saving model')
    model.save("saved_model/vae-default")