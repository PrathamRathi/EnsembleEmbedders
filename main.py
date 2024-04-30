from src.midi_preprocess import get_midi_paths 
from src.midi_utils import get_data_from_midi, get_midi_from_data
from net.autoencoder import Autoencoder, LossAccuracyCallback
import numpy as np

data_path = 'data/data/lyricsMidisP0'
# data_path = 'data/lyricsMidisP0'
output_path = 'output/'

if __name__ == "__main__":
    midis = get_midi_paths(data_path, depth=2)
    midis = midis[0][:10] # data is split into 5 sections, pick the first one.
    print('Total files: {}'.format(len(midis)))
    
    max_length = get_data_from_midi(midis[0]).shape[1]
    for i in range(len(midis)):
        print('Processing file {}.'.format(i))
        melody_lyrics_array = get_data_from_midi(midis[i], verbose=False)
        curr_length = melody_lyrics_array.shape[1]
        if (max_length < curr_length):
            max_length = curr_length
        # midi_reconstructed = get_midi_from_data(melody_lyrics_array, tempo = 240)
        # output_filename = output_path + 'result' + str(i) +'.mid'
        # midi_reconstructed.write(output_filename)

    data =[]
    for i in range(len(midis)):
        mel_array = get_data_from_midi(midis[i], verbose=False)
        curr_length = mel_array.shape[1]
        if curr_length < max_length:
            padding = max_length - curr_length
            mel_array = np.pad(mel_array, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            data.append(mel_array)

    data.append(melody_lyrics_array)
    [print(x.shape) for x in data]