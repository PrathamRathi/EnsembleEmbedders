import glob
import os
import pretty_midi
import numpy as np
import joblib

original_dir = 'lmd_full/'
lyrics_dir = 'lyricsMIDIS/'

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
    split_paths = np.array_split(paths, 5)
    return split_paths

def midi_to_txt(midi_files, txt_file):
    for midi in midi_files:
        pm = pretty_midi.PrettyMIDI(midi)
        lyrics = pm.lyrics
        for lyric in lyrics:
            with open(txt_file, 'w') as f:
                f.write(f"{lyric.text}\n")


# split = get_midi_paths(original_dir, depth=2)

# lyrics_counts = joblib.Parallel(n_jobs=10, verbose=0)(
#     joblib.delayed(move_midis)(chunk, lyrics_dir)
#     for chunk in split
# )

lyrics_paths = get_midi_paths(lyrics_dir)
joblib.Parallel(n_jobs=10, verbose=0)(
    joblib.delayed(midi_to_txt)(chunk, 'vocab/' + str(count) + '.txt')
    for count, chunk in enumerate(lyrics_paths)
)
