from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.autoencoder import Autoencoder, LossAccuracyCallback
import numpy as np
import tensorflow as tf

data_path = 'data/data/lyricsMidisP0'
# data_path = 'data/lyricsMidisP0'
output_path = 'output/'

if __name__ == "__main__":
    #load midis
    midis = get_midi_paths(data_path, depth=2)
    n = 1
    midis = midis[0][:n] # data is split into 5 sections, pick the first one.
    print('Total files: {}'.format(len(midis)))

    midi_0 = get_data_from_midi(midis[0])
    max_length = midi_0.shape[1]
    max_pitches = midi_0.shape[0] #128 pitches
    for i in range(len(midis)):
        print('Processing file {}.'.format(i))
        melody_lyrics_array = get_data_from_midi(midis[i])
        curr_length = melody_lyrics_array.shape[1]
        if (max_length < curr_length):
            max_length = curr_length
        # midi_reconstructed = get_midi_from_data(melody_lyrics_array, tempo = 240)
        # output_filename = output_path + 'result' + str(i) +'.mid'
        # midi_reconstructed.write(output_filename)

    data = []
    for i in range(len(midis)):
        mel_array = get_data_from_midi(midis[i])
        curr_length = mel_array.shape[1]
        if curr_length < max_length:
            padding = max_length - curr_length
            mel_array = np.pad(mel_array, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        data.append(mel_array)

    # [print(x.shape) for x in data]
    x_train = tf.constant(data)
    print("main.py, data shape:", x_train.shape)

    model = Autoencoder(song_length= max_length,
                        instrument_units= 1,
                        pitch_units=max_pitches
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

