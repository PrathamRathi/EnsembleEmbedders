import numpy as np
import pretty_midi as pm
from time import sleep
import random


def name2num(name):
    return pm.instrument_name_to_program(name)

def num2class(n):
    return pm.program_to_instrument_class(n)

def get_chroma_from_midi(midi_path, verbose=False):
    if verbose:
        print("Extracting chroma from midi file: {}".format(midi_path))
        print()
    mid = pm.PrettyMIDI(midi_path)
    BPM = mid.get_tempo_changes()[1][0]
    sample_length = (60.0 / 8.0) / BPM
    start = 0 # TODO: Fix this
    end = start + 320.0 * sample_length
    chroma_rolls = np.zeros((3, 12, 320))
    i = 0
    for instr in mid.instruments:
        if i > 2:
            break
        chroma_roll = instr.get_chroma(times=np.arange(start, end, sample_length)).astype(bool) + 0
        if chroma_roll.shape[1] == 321:
            chroma_roll = chroma_roll[:, :-1]
        if np.max(chroma_roll):
            chroma_rolls[i, :, :] = chroma_roll
            i += 1
    return chroma_rolls

def get_midi_from_chroma(chroma_rolls, tempo, verbose=False):
    if verbose:
        print('Creating MIDI from tensor of shape {}'.format(chroma_rolls.shape))

    # Convert chroma rolls to bool
    chroma_rolls = chroma_rolls > 0.5

    midi = pm.PrettyMIDI()
    # Synthesize all melodies with a piano
    piano_num = name2num('Acoustic Grand Piano')
    instruments = [pm.Instrument(program=piano_num),
                   pm.Instrument(program=piano_num),
                   pm.Instrument(program=piano_num)]
    melodies = [chroma_rolls[0, :, :],
                chroma_rolls[1, :, :],
                chroma_rolls[2, :, :]]

    # Compute the length of a beat using the given tempo
    mins_per_beat = 1 / tempo
    secs_per_8th_beat = mins_per_beat * (60 / 8)

    for i in range(3):
        # Add all melodies back into the midi
        for row in range(melodies[i].shape[0]):
            # Compute the pitch
            note_pitch = row
            col = 0
            if verbose:
                print('Processing note number/row: {}'.format(row))
            # For each timestep
            while col < melodies[i].shape[1]:
                note_vel = melodies[i][row, col]
                # If there is no note...
                if note_vel == 0:
                    col += 1
                    continue
                # Otherwise...
                note_start = col * secs_per_8th_beat
               
                while col < melodies[i].shape[1] and melodies[i][row, col]:
                    col += 1
                note_end = col * secs_per_8th_beat
                
                if note_vel:
                    note = pm.Note(80, note_pitch + 12 * (i + 4), note_start, note_end)
                    instruments[i].notes.append(note)
        midi.instruments.append(instruments[i])

    return midi
    
if __name__ == '__main__':

    # TEST FUNCTIONS
    # chroma_rolls = get_chroma_from_midi('data/Dancing Queen.mid', verbose=True)
    # print('chroma_rolls.shape: {}'.format(chroma_rolls.shape))
    # midi = get_midi_from_chroma(chroma_rolls, 120)
    # midi.write('dancing-queen-from-chroma.mid')
    MELODY_HARMONY_INSTRUMENTS = ['Piano', 'Synth Lead', 'Guitar', 'Brass', 'Organ']
    BASS_INSTRUMENTS = ['Bass']

    midi_paths = np.load("midi_paths.npy")
    print("MidiPathsArrayShape:", midi_paths.shape)

    base_save_path = "chroma_rolls_batch_"

    # chroma_rolls = np.zeros((5000, 3, 12, 320)).astype(int)
    chroma_rolls = []
    save_index = -1
    for n in range(0, len(midi_paths)):
        local_n = (n % 5000)
        if n % 100 == 0:
            print("n:", n)
        try:
            path = midi_paths[n]
            mid = pm.PrettyMIDI(path)
            BPM = mid.get_tempo_changes()[1][0]
            sample_length = (60.0 / 8.0) / BPM
            interval_end = 320.0 * sample_length
            i = 0
            # Evan, I think we can try to do this in a slightly inefficient way--cuz
            #    hopefully it'll still be fast enough to run overnight.

            # Try to get a melody, harmony, and bass instrument
            candidate_melody_harmony_instruments = [instr for instr in mid.instruments if num2class(instr.program) in MELODY_HARMONY_INSTRUMENTS]
            candidate_bass_instruments = [instr for instr in mid.instruments if num2class(instr.program) in BASS_INSTRUMENTS]

            # Use the highest melody_harmony instrument as the melody
            highest = None
            for instr in candidate_melody_harmony_instruments:
                pass
                # TODO: Get highest instrument

            # Use a random (different) melody_harmony instrument as the harmony
            # TODO: Remove the instrument picked as the melody, then choose a random instrument

            # Use a random bass instrument as the bass
            # TODO: Use random candidate bass instrument

            # If we still don't have 3 instruments, look through the rest and take anything that works
            for instr in mid.instruments:
                if i > 2:
                    break
                chroma_roll = instr.get_chroma(times=np.arange(0, interval_end, sample_length)).astype(bool) + 0
                if chroma_roll.shape[1] == 321:
                    chroma_roll = chroma_roll[:, :-1]
                # If the instrument has notes, add it
                if np.max(chroma_roll):
                    chroma_rolls[local_n, i, :, :] = chroma_roll
                    i += 1
        except Exception as e:
            print("Error:", e)
        if (n + 1) % 5000 == 0:
            save_index += 1
            save_path = base_save_path + str(save_index)
            np.save(save_path, chroma_rolls)
            sleep(2)
            print("NextBatch")
            chroma_rolls = np.zeros((5000, 3, 12, 320)).astype(int)

    chroma_rolls_all = None
    for i in range(0, 17):
        path_to_load = "chroma_rolls_batch_" + str(i) + ".npy"
        print("path_to_load:", path_to_load)
        if chroma_rolls_all is None:
            chroma_rolls_all = np.load(path_to_load)
        else:
            chroma_rolls_all = np.concatenate((chroma_rolls_all, np.load(path_to_load)), axis=0)
    print("TotalShape:", chroma_rolls_all.shape)
    print("dtype:", chroma_rolls_all.dtype)
    np.save("chroma_rolls_all", chroma_rolls_all)
    print("Saved chroma_rolls_all")
