from src.midi_utils import get_midi_from_data_eric, get_midi_from_data_evan, get_midi_from_data
import numpy as np
import tensorflow as tf
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=int, required = True, help = "0 for 128 pitches, 1 for 12")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if (args.f == 0):
        processing = get_midi_from_data_eric
    elif (args.f == 1):
        processing = get_midi_from_data_evan
    
    model_path = "saved_model/default.keras"
    model = tf.keras.models.load_model(model_path)
    model.summary()
    test_midi_file = "data/data/lyricsMidisP0/"
    test_midi_processed = processing(test_midi_file)
    model_midi_processed = model.predict(test_midi_processed)
    reconstructed_midi = get_midi_from_data(model_midi_processed)
