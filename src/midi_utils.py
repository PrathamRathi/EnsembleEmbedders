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

def get_tempo(midi, verbose=False):
    # Get all tempo changes throughout the song
    times, tempos = midi.get_tempo_changes()
    if verbose:
        print("Extracted {} tempos".format(len(tempos)))

    # Simply use the first tempo as the tempo for the whole song, ignore changes
    tempo = tempos[0]
    if verbose:
        print("using first tempo: {} BPM".format(tempo))
        print()
    return tempo


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
        print()
    return bass_instrument

def get_data_from_midi(midi_path, verbose=False):
    if verbose:
        print("Extracting data from midi file: {}".format(midi_path))
        print()
    midi = pm.PrettyMIDI(midi_path)
    melody_instr = get_melody_instr(midi, verbose=verbose)
    # bass_instr = get_bass_instr(midi, verbose=verbose)

    # Extract the tempo so we can sample piano roll
    #   by beats instead of seconds.
    # We will sample by 16th notes, so that every
    #   column represents a 16th note.
    BPM = get_tempo(midi, verbose=verbose)

    # In get_piano_roll, columns are spaced apart by 1./fs
    #   We want them to be spaced by 1/16th of a beat
    MPB = 1 / BPM # minutes per beat
    SPB = 60 * MPB # seconds per beat
    sample_length = SPB / 16 # seconds per 16th of a beat
    sampling_rate = 1 / sample_length
    
    melody_roll = melody_instr.get_piano_roll(fs=sampling_rate)
    # bass_roll = bass_instr.get_piano_roll(fs=sampling_rate)
    if verbose:
        print("Shape of tensors: {}".format(melody_roll.shape))

    # Get the lyrics and associate them to nearest 1/16th note
    lyrics_roll = list(np.zeros(melody_roll.shape[1], dtype=str)) # single row along time axis
    for lyric in midi.lyrics:
        # Remove unnecessary characters and whitespace
        lyric_text = lyric.text.lower()
        lyric_text = lyric_text.replace('\n', '')
        lyric_text = lyric_text.replace('\r', '')
        lyric_text = lyric_text.replace('\t', '')
        lyric_text = lyric_text.strip()
        lyric_text = lyric_text.strip('.,-_')
        if lyric_text == "":
            continue
        # Compute closest index into 1/16th note columns
        best_index = int(np.floor(lyric.time / sample_length))
        lyrics_roll[best_index] = lyric_text

    # Combine the melody_roll with the lyrics_roll
    lyrics_roll = np.array(lyrics_roll).reshape(1, melody_roll.shape[1])
    melody_lyrics_array = np.concatenate([melody_roll, lyrics_roll], axis=0)
    
    return melody_lyrics_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/Dancing Queen.mid", help="path to midi.")
    parser.add_argument("-v", "--verbose", type=bool, default=False, help="whether or not to print progress messages.")
    options = parser.parse_args()
    melody_lyrics_array = get_data_from_midi(options.input, verbose=True)
