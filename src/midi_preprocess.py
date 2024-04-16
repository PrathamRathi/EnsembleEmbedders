import glob
import os
import pretty_midi
import numpy as np
import joblib
import argparse

def move_midis(original_paths, new_dir):
    corrupted_count = 0
    lyrics_count = 0
    for path in original_paths:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            if pm.lyrics:
                lyrics_count += 1
                split = path.split('/')
                os.rename(path, new_dir + split[-1])
                lyrics_count += 1
        except:
            corrupted_count += 1  
    print(f'Found {corrupted_count} corrupted files')
    return lyrics_count


def get_midi_paths(midi_dir, depth=1,split_count=5):
    wildcard = '*.mid'
    if depth == 2:
        wildcard = '**/*.mid'
    paths = []
    for path in glob.iglob(midi_dir + wildcard, recursive=True):
        paths.append(path)  
    paths = np.array(paths) 
    split_paths = np.array_split(paths, split_count)
    return split_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type = str, default = "data/", help="path to the data folder containing the entire lakh dataset.")
    parser.add_argument("--lyrics", type = str, default = "data/lyricsMIDIS/", help="path to where lyrical MIDIs should be written")
    options = parser.parse_args()
    original_dir = options.original
    lyrics_dir = options.lyrics
    split = get_midi_paths(original_dir, depth=2)
    lyrics_counts = joblib.Parallel(n_jobs=10, verbose=0)(
        joblib.delayed(move_midis)(chunk, lyrics_dir)
        for chunk in split
    )
