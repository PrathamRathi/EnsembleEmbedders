'''
import numpy as np
import pretty_midi as pm
from time import sleep

midi_paths = np.load("midi_paths.npy")
print("MidiPathsArrayShape:", midi_paths.shape)

base_save_path = "chroma_rolls_batch_"

chroma_rolls = np.zeros((5000, 3, 12, 160)).astype(int)
save_index = -1
for n in range(0, len(midi_paths)):
    local_n = (n % 5000)
    if n % 100 == 0:
        print("n:", n)
    try:
        path = midi_paths[n]
        mid = pm.PrettyMIDI(path)
        BPM = mid.get_tempo_changes()[1][0]
        sample_length = (60.0 / 4.0) / BPM
        interval_end = 160.0 * sample_length
        i = 0
        for instr in mid.instruments:
            if i > 2:
                break
            chroma_roll = instr.get_chroma(times=np.arange(0, interval_end, sample_length)).astype(bool) + 0
            if chroma_roll.shape[1] == 161:
                chroma_roll = chroma_roll[:, :-1]
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
        chroma_rolls = np.zeros((5000, 3, 12, 160)).astype(int)

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
'''