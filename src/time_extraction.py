import pretty_midi as pm
import numpy as np

# get midi object from midi file
mid = pm.PrettyMIDI("data/Dancing Queen.mid")

# set hyperparameters
num_time_intervals = 200
num_instruments_to_use = 3
min_tot_notes_per_instrument = 20 # skip over the instrument if it has too few total notes

# initialize the list which will store note-and-word content for each time interval
time_intervals = [([[] for _ in range(num_instruments_to_use)] + [""]) for _ in range(num_time_intervals)]

# add the notes to the correct buckets
i = 0
for instrument in mid.instruments:
    if len(instrument.notes) >= min_tot_notes_per_instrument:
        for note in instrument.notes:
            rounded_note_timestamp = int(0.75 * note.start + 0.25 * note.end) # start is a bit more important than end (?)
            if rounded_note_timestamp < num_time_intervals:
                time_intervals[rounded_note_timestamp][i].append(note.pitch)
        i += 1
    if i >= num_instruments_to_use:
        break

# add the words to the correct buckets
for word in mid.lyrics:
    rounded_word_timestamp = int(word.time)
    if rounded_word_timestamp < num_time_intervals:
        time_intervals[int(word.time)][-1] += ("_" + word.text)

# print the time interval content as rows
for time_interval in time_intervals:
    print(time_interval)