import pretty_midi as pm
import numpy as np
import argparse

# For melody, base, and drums--rank instruments in order of likelihood
# See all 128 MIDI instruments here--https://en.wikipedia.org/wiki/General_MIDI
# We will use classes rather than exact instruments

# NOTE: "In GM standard MIDI files, channel 10 is reserved for percussion instruments only."
MELODY_INSTRUMENT_RANKING = ['Piano', 'Synth Lead', 'Guitar', 'Brass', 'Organ']
BASS_INSTRUMENT_RANKING = ['Bass', 'Synth Pad']
DRUM_INSTRUMENT_RANKING = ['Percussive']

def num2class(n):
    return pm.program_to_instrument_class(n)

def num2name(n):
    return pm.program_to_instrument_name(n)

def get_midi(data):
    return None

def get_melody_instr(midi, verbose=False):        
    for instr_class in MELODY_INSTRUMENT_RANKING:
        # Find list of instruments matching each class
        melody_instruments = [instr for instr in midi.instruments if num2class(instr.program) == instr_class]

        # If we found any instruments, break
        if len(melody_instruments) > 0:
            break
    
    if (verbose):
        print('Extracted {} candidate melody instruments of class {}:'.format(len(melody_instruments), instr_class))
        print([num2name(instr.program) for instr in melody_instruments])

    # From the list of instruments that matched, take the one with the most notes
    melody_instrument = melody_instruments[0]
    max_num_notes = -1
    for instr in melody_instruments:
        notes = instr.notes
        num_notes = len(notes)
        if num_notes > max_num_notes:
            max_num_notes = num_notes
            melody_instrument = instr

    if verbose:
        print("using melody instrument: {}".format(num2name(melody_instrument.program)))
        print()
    return melody_instrument

def get_bass_instr(midi, verbose=False):
    for instr_class in BASS_INSTRUMENT_RANKING:
        # Find list of instruments matching each class
        bass_instruments = [instr for instr in midi.instruments if num2class(instr.program) == instr_class]

        # If we found any instruments, break
        if len(bass_instruments) > 0:
            break

    if (verbose):
        print('Extracted {} candidate bass instruments of class {}:'.format(len(bass_instruments), instr_class))
        print([num2name(instr.program) for instr in bass_instruments])

    # From the list of instruments that matched, take the one with the most notes
    bass_instrument = bass_instruments[0]
    max_num_notes = -1
    for instr in bass_instruments:
        notes = instr.notes
        num_notes = len(notes)
        if num_notes > max_num_notes:
            max_num_notes = num_notes
            bass_instrument = instr

    if verbose:
        print("using bass instrument: {}".format(num2name(bass_instrument.program)))
    return bass_instrument

def get_data(midi_path, verbose=False):
    if verbose:
        print("Extracting data from midi file: {}".format(midi_path))
        print()
    midi = pm.PrettyMIDI(midi_path)
    melody_instr = get_melody_instr(midi, verbose=verbose)
    bass_instr = get_bass_instr(midi, verbose=verbose)
    
    melody_roll = melody_instr.get_piano_roll()
    bass_roll = bass_instr.get_piano_roll()
    lyrics = midi.lyrics

    # To integrate with Evan's stuff:
    # melody_roll = quantize(melody_roll)
    # lyrics = quantize(lyrics)

    return melody_roll, bass_roll, lyrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default = "data/", help="path to the data folder containing midis.")
    parser.add_argument("--example", type = str, default = "data/Dancing Queen.mid", help="path to one midi.")
    options = parser.parse_args()
    melody_roll, bass_roll, lyrics = get_data(options.example, verbose=True)
    # print(roll.shape)
    # print()
    # print(lyrics)
