import pretty_midi as pm
import numpy as np

def num2name(n):
    return pm.program_to_instrument_name(n)

def name2num(name):
    return pm.instrument_name_to_program(name)

def combine_midis(midi_paths):
    input_midis = [pm.PrettyMIDI(mp) for mp in midi_paths]
    output_midi = pm.PrettyMIDI()

    # Take first instrument of each input midi and add each to the output midi
    for in_midi in input_midis:
        instr = in_midi.instruments[0]
        output_midi.instruments.append(instr)

    return output_midi


if __name__ == '__main__':
    combine_dir = 'midis_to_combine/'
    output_midi = combine_midis([combine_dir + fname for fname in [
        'minor-bass.mid',
        'minor-chords.mid',
        'minor-melody.mid'
    ]])
    output_midi.write(combine_dir + 'combined.mid')