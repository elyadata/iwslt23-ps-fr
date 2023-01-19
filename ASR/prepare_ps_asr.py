#!/usr/bin/env python3
"""
Pashto-French data processing.

Author
------
Haroun Elleuch 2023
"""

import json
import os
from tqdm import tqdm


def write_json(json_file_name, data):
    with open(json_file_name, encoding="utf-8", mode="w") as output_file:
        json.dump(
            data,
            output_file,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )
    print("Saved: ",json_file_name)


def generate_json(dataset_folder, split, max_duration):
    print(f"Generating JSON manifest for {split} split...")
    output_json = dict()
    ignored_segments_short = list()
    ignored_segments_long = list()
    SAMPLING_RATE = 16000
    # Loading speech segments descriptions of the split
    with open(os.path.join(dataset_folder,"txt",f"{split}.json")) as file:
        description = json.load(file)
        
    # Sorting the descriptions by filename field
    description = sorted(description, key=lambda d: d['filename'])
        
    files = get_files(description)
    
    # creating a dict to keep track of the last used index for each file segment
    segment_counters = dict()
    for file in files:
        segment_counters[file] = 0
        
    # generating a unique segment ID
    utt_ids = set()
    for utt in tqdm(description):
        filename = utt["filename"].split(".wav")[0]
        segment_counters[filename] += 1
        utt_id = f"{filename}_id_{segment_counters[filename]}"
                
        assert utt_id not in utt_ids
        utt_ids.add(utt_id)
        
        # Creating a new segment entry to be saved in the output_json
        entry = dict()
        entry["path"] = os.path.join(dataset_folder, "wav", utt["filename"])
        entry["transcription"] = utt["src"]
        entry["start"] = int(utt["start"]*SAMPLING_RATE)
        entry["stop"] = int(utt["end"]*SAMPLING_RATE)
        entry["duration"] = entry["stop"] - entry["start"]
        
        # Adding segment to the output JSON
        # ignoring small and large samples due to cuda memory errors
        if entry["duration"] <= 1 * SAMPLING_RATE:
            ignored_segments_short.append(utt)
        elif entry["duration"] >= max_duration * SAMPLING_RATE:
            ignored_segments_long.append(utt)
        else:
            output_json[utt_id] = entry
    # Integrity check: same number of segments
    # assert len(output_json) == len(description)
    print(f"Ignored {len(ignored_segments_short)} segments for being too short and {len(ignored_segments_long)} for being too long.")
    
    return output_json, ignored_segments_short, ignored_segments_long

def get_files(description: list) -> set:
    """ Gets a set of filenames used in the dataset split
    Args:
        description (dict): A dictionnary containing the loaded JSON description file contents

    Returns:
        set: All files referenced in the JSON contents
    """
    files = set()
    for segment in description:
        files.add(segment["filename"].split(".wav")[0])
    return files

def data_proc(dataset_folder, output_folder, max_duration=179):
    """
    Prepare json files for the Pashto speech to french text datatset.

    Arguments:
    ----------
        dataset_folder (str) : path for the dataset root folder
        output_folder (str) : path where we save the new json files.
    """

    try:
        os.mkdir(output_folder)
    except OSError:
        print(
            "Tried to create " + output_folder + ", but folder already exists."
        )

    for split in ["train", "dev", "test"]:
        try:
            output_json, too_short, too_long = generate_json(dataset_folder, split, max_duration)     
        except FileNotFoundError:
            print(f"Could not find description file for {split} split.")   
            
        write_json(os.path.join(output_folder, f"{split}.json"), output_json)
        
        if len(too_short) > 0:
            write_json(os.path.join(output_folder, f"{split}_bad_segments_short.json"), too_short)
        
        if len(too_long) > 0:
            write_json(os.path.join(output_folder, f"{split}_bad_segments_long.json"), too_long)

