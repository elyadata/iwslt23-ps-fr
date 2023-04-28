#!/usr/bin/env python3
"""
Pashto-French data processing.

Author
------
Haroun Elleuch 2023
"""

import json
import logging
import os
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
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
    print("Saved: ", json_file_name)


def generate_json(dataset_folder, manifest_file_name, max_duration, min_duration):
    output_json = dict()
    ignored_segments_short = list()
    ignored_segments_long = list()
    SAMPLING_RATE: int = 16000
    # Loading speech segments descriptions of the split
    with open(os.path.join(dataset_folder, manifest_file_name)) as file:
        description = json.load(file)

    # generating a unique segment ID
    utt_ids = set()
    for utt in tqdm(description):
        segment = description[utt]

        assert utt not in utt_ids  # Ensuring that the segment has not been preprocessed previously
        utt_ids.add(utt)

        # Creating a new segment entry to be saved in the output_json
        entry = dict()
        entry["path"] = os.path.join(dataset_folder, "wav", segment["filename"])
        entry["start"] = int(segment["start"] * SAMPLING_RATE)
        entry["stop"] = int(segment["end"] * SAMPLING_RATE)
        entry["duration"] = int(segment["duration"] * SAMPLING_RATE)

        # Adding segment to the output JSON
        # ignoring small and large samples due to cuda memory errors
        if segment["duration"] <= min_duration:
            ignored_segments_short.append(segment)
        elif segment["duration"] >= max_duration:
            ignored_segments_long.append(segment)
        else:
            output_json[utt] = entry

    print(f"Ignored {len(ignored_segments_short)} segments for being too short \
        and {len(ignored_segments_long)} for being too long.")

    return output_json, ignored_segments_short, ignored_segments_long


def data_proc(input_json_file_path, output_json_file_path, dataset_folder, max_duration, min_duration):
    """
    Prepare json files for the Pashto speech to french text datatset.

    Arguments:
    ----------
        dataset_folder (str) : path for the dataset root folder
        output_folder (str) : path where we save the new json files.
    """

    output_json, too_short, too_long = generate_json(dataset_folder, "DW_Pashto.json", max_duration, min_duration)

    write_json(output_json_file_path, output_json)

    base_file_name = output_json_file_path.split('.json')[0]
    if len(too_short) > 0:
        write_json(base_file_name + "_bad_segments_short.json", too_short)

    if len(too_long) > 0:
        write_json(base_file_name + "_bad_segments_long.json", too_long)


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

        # creates a logger
    logger = logging.getLogger(__name__)
    data_proc(
        input_json_file_path=hparams["unprepared_train_json_file"],
        output_json_file_path=hparams["train_json"],
        dataset_folder=hparams["unlabelled_data_folder"],
        max_duration=hparams["avoid_if_longer_than"],
        min_duration=hparams["avoid_if_shorter_than"]
    )
