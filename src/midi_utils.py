import pretty_midi
import numpy as np
import argparse

def get_midi(data):
    return None

def get_melody_instr(midi):
    # Assume the piano instrument with the most notes is the melody
    piano_instruments = [instr for instr in midi.instruments if pretty_midi.program_to_instrument_class(instr.program) == 'Piano']
    
    melody_instrument = piano_instruments[0]
    max_num_notes = -1

    for instr in piano_instruments:
        notes = instr.notes
        num_notes = len(notes)
        if num_notes > max_num_notes:
            max_num_notes = num_notes
            melody_instrument = instr

    return melody_instrument

def get_data(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    melody_instr = get_melody_instr(midi)
    
    melody_roll = melody_instr.get_piano_roll()
    lyrics = midi.lyrics

    return melody_roll, lyrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default = "data/", help="path to the data folder containing midis.")
    parser.add_argument("--example", type = str, default = "data/Dancing Queen.mid", help="path to one midi.")
    options = parser.parse_args()
    roll, lyrics = get_data(options.example)
    print(roll)
    print()
    print(lyrics)
