from net.variational_autoencoder import VAE
import numpy as np
import tensorflow as tf
import os
import argparse
import json

<<<<<<< HEAD
#-epochs 100 -lr 3e-5 -file chroma_rolls_all.npy
# data_path = 'data/data/lyricsMidisP0'
=======
>>>>>>> refs/remotes/origin/main
data_path = 'data/data/lyricsMidisP0'
output_path = 'output/'
preprocessed_folder_path = "preprocessed/"
default_preprocessed_path = 'preprocessed/all_chunks' # all_chunks is the filename of the .npy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, required = True, help = "epochs")
    parser.add_argument("-lr", type=float, required = True, help = "learning rate")
    parser.add_argument("-file", type=str, required = True, help = "preprocessed .npy file path", default= "default")
<<<<<<< HEAD
    parser.add_argument("-name", type=str, required = True, help = "name for the saved model", default= "default")
=======
    parser.add_argument("-model", type=str, help = "type of model to use (dense, cdc, etc.)", default= "dense")
>>>>>>> refs/remotes/origin/main
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
    instrument_units = 3
    pitch_units = 12
    song_length = x_train.shape[-1]
    print(f"Loading preprocessed data from file {preprocessed_path}. Shape: {x_train.shape}")

    # Choose model for training
    model = None
    name = args.model + '-'
    if args.model == 'dense':
        model = VAE(song_length= song_length,instrument_units= instrument_units,pitch_units= pitch_units,
                    learning_rate= args.lr,epochs=args.epochs,
                            hidden_dim=512,latent_size=16)
    elif args.model == 'cdc':
        # TODO: set model to conv-deconv
        model = None
    
    # Train model
    model.compile(optimizer = model.optimizer,)
    model.build(input_shape = (1,instrument_units, pitch_units, song_length))
    model.summary()
<<<<<<< HEAD
    model.fit(x_train, 
              x_train,
                epochs=model.epochs,
              batch_size = 10, #TODO: changing batch size breaks training
              validation_split = 0.2
    )
    print('Saving model')
    saved_model_path = f'saved_model/{args.name}_e{args.epochs}_lr{args.lr}.keras'
    model.save(saved_model_path)
=======
    history = model.fit(x_train, x_train,epochs=model.epochs,batch_size = 32 if model.epochs > 300 else 10,validation_split = 0.2)

    # Save model and training history
    MODEL_DIR = 'saved_model/'
    HIST_DIR = 'saved_model/history/'
    print('Saving model')
    model.save(MODEL_DIR + name + 'vae')
    print('Saving model history')
    out_file = open(HIST_DIR + name + 'vae.json', "w") 
    json.dump(history.history, out_file)
>>>>>>> refs/remotes/origin/main
