import numpy as np
import pretty_midi as pm
from time import sleep

L0 = ["Piano", "Synth Lead", "Organ", "Guitar", "Strings", "Reed", "Pipe", "Brass"]
L1 = ["Bass", "Ensemble"]

midi_paths = np.load("midi_paths.npy")
num_midis_to_check = len(midi_paths)
base_save_path = "chroma_rolls_batch_"

chroma_rolls = []
save_index = -1
n = 0
while n < num_midis_to_check:
    if n % 50 == 0:
        print("Checking Midi", n)
        print("ChromaRollsListLength:", len(chroma_rolls))
    try:
        mid = pm.PrettyMIDI(midi_paths[n])
        BPM = mid.get_tempo_changes()[1][0]
        sample_length = (60.0 / 8.0) / BPM
        start = np.random.randint(0, 20) * 8.0
        end = start + 320.0 * sample_length
        L0_candidates, L1_candidates, L2_candidates = [], [], []
        for instr in mid.instruments:
            instr_class = pm.program_to_instrument_class(instr.program)
            if instr_class in L0:
                L0_candidates.append(instr)
            elif instr_class in L1:
                L1_candidates.append(instr)
            else:
                pass
        if (len(L0_candidates) > 1) and (len(L1_candidates) > 0):
            instr_to_use = []
            L0_candidates_avg_pitches = [np.mean([note.pitch for note in instr.notes]) for instr in L0_candidates]
            zipped_L0_pairs = zip(L0_candidates, L0_candidates_avg_pitches)
            sorted_L0_pairs = sorted(zipped_L0_pairs, key = lambda pair: pair[1])
            instr_to_use.append(sorted_L0_pairs[-1][0])
            instr_to_use.append(sorted_L0_pairs[-2][0])
            L1_candidates_avg_pitches = [np.mean([note.pitch for note in instr.notes]) for instr in L1_candidates]
            zipped_L1_pairs = zip(L1_candidates, L1_candidates_avg_pitches)
            sorted_L1_pairs = sorted(zipped_L1_pairs, key = lambda pair: pair[1])
            instr_to_use.append(sorted_L1_pairs[0][0])
            chroma_rolls.append(np.zeros((3, 12, 320)))
            for i in range(3):
                chroma_roll = instr_to_use[i].get_chroma(times=np.arange(start, end, sample_length)).astype(bool) + 0
                if chroma_roll.shape[1] == 321:
                    chroma_roll = chroma_roll[:, :-1]
                chroma_rolls[-1][i, :, :] = np.copy(chroma_roll)
    except Exception as e:
        print(e, midi_paths[n])
    if len(chroma_rolls) > 999:
        save_index += 1
        chroma_rolls = np.array(chroma_rolls)
        np.save(base_save_path + str(save_index), chroma_rolls)
        sleep(2)
        print("Saved", str(save_index) + "th", "batch with shape", chroma_rolls.shape)
        chroma_rolls = []
        print("ChromaRollsList has been reset")
        print("ChromaRollsListLength:", len(chroma_rolls))
    n += 1