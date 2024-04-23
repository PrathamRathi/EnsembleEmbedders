import pretty_midi
import numpy as np
import argparse
from collections import Counter
import os
import string
from operator import itemgetter

def format_lyric(lyric):
    lyric_text = lyric.text.lower()
    lyric_text = lyric_text.replace('\n', '')
    lyric_text = lyric_text.replace('\r', '')
    lyric_text = lyric_text.replace('\t', '')
    lyric_text = lyric_text.strip()
    lyric_text = lyric_text.strip('.,-_')
    lyric_text = lyric_text.translate(str.maketrans('', '', string.punctuation))
    lyric_split = lyric_text.split(' ')
    return lyric_split
        
def midi_to_lyrics(midi_path):
    text_list = []
    pm = pretty_midi.PrettyMIDI(midi_path)
    lyrics = pm.lyrics
    for l in lyrics:
        line = format_lyric(l)
        text_list += line
    return text_list

def dir_to_vocab(midi_dir):
    files = os.listdir(midi_dir)
    frequencies = {}
    for i, file in enumerate(files):
        if i % 500 == 0:
            print(f"On file {i}")
        path = midi_dir + '/' + file
        lyrics_list = midi_to_lyrics(path)
        midi_frequencies = Counter(lyrics_list)
        frequencies.update(midi_frequencies)
    
    for word in list(frequencies):
        if not word.isalpha():
            del frequencies[word]
    return frequencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="data/lyricsMidisP", help="path to data subfolder.")
    options = parser.parse_args()

    # Extract training data from midi file
    overall_frequencies = {}
    for i in range(1):
        dir = options.input + str(i)
        print(f"starting to process directory {dir}")
        curr_frequencies = dir_to_vocab(dir)
        overall_frequencies.update(curr_frequencies)

    print(len(overall_frequencies))

    res = dict(sorted(overall_frequencies.items(), key=itemgetter(1), reverse=True)[:10000])
    print("The top 10000 value pairs are " + str(res))


