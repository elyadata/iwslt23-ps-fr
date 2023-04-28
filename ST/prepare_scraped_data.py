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


def generate_json(
        scraped_dataset_folder,
        official_dataset_folder,
        manifest_file_path,
        max_duration,
        min_duration,
        mode,
        sample_rate=16000,
):
    output_json = dict()
    ignored_segments_short = list()
    ignored_segments_long = list()
    # Loading speech segments descriptions of the split
    with open(manifest_file_path) as file:
        segments = json.load(file)

    # generating a unique segment ID
    utt_ids = set()
    for segment in tqdm(segments):
        segment_id = segment["segment_id"]

        if segment["dataset"] == "official_train_data":
            path_prefix = official_dataset_folder
        elif (
                segment["dataset"] == "dw_pashto_airbus_translation"
                or segment["dataset"] == "dw_pashto_google_translation"
        ):
            path_prefix = scraped_dataset_folder
        else:
            logger.error("Cannot identify dataset of orgin. Existing...")
            break

        assert (
                segment_id not in utt_ids
        )  # Ensuring that the segment has not been preprocessed previously
        utt_ids.add(segment_id)

        # Creating a new segment entry to be saved in the output_json
        entry = dict()
        entry["path"] = os.path.join(path_prefix, segment["path"])
        entry["start"] = segment["start"]
        entry["stop"] = segment["stop"]
        entry["duration"] = segment["duration"]

        if mode.lower() == "st":
            entry["translation"] = segment["translation"]
        elif mode.lower() == "asr":
            entry["transcription"] = segment["transcription"]
        elif mode.lower() == "joint":
            entry["translation"] = segment["translation"]
            entry["transcription"] = segment["transcription"]
        else:
            raise NotImplementedError(f"No mode {mode} implemented for this data preparation step. Exiting...")

        # Adding segment to the output JSON
        # ignoring small and large samples due to cuda memory errors
        if segment["duration"] <= (min_duration * sample_rate):
            ignored_segments_short.append(segment)
        elif segment["duration"] >= (max_duration * sample_rate):
            ignored_segments_long.append(segment)
        else:
            output_json[segment_id] = entry

    print(
        f"Ignored {len(ignored_segments_short)} segments for being too short \
        and {len(ignored_segments_long)} for being too long."
    )

    return output_json, ignored_segments_short, ignored_segments_long


def prepare_dw_pashto(
        input_json_file_path,
        output_json_file_path,
        scraped_dataset_folder,
        official_dataset_folder,
        max_duration,
        min_duration,
        mode="st",
):
    """
    Prepare json files for the Pashto speech to french text datatset.
    """
    output_json, too_short, too_long = generate_json(
        scraped_dataset_folder,
        official_dataset_folder,
        input_json_file_path,
        max_duration,
        min_duration,
        mode,
    )

    write_json(output_json_file_path, output_json)

    base_file_name = output_json_file_path.split(".json")[0]
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
    prepare_dw_pashto(
        input_json_file_path=hparams["unprepared_train_json_file"],
        output_json_file_path=hparams["train_json"],
        scraped_dataset_folder=hparams["unlabelled_data_folder"],
        official_dataset_folder=hparams["official_data_folder"],
        max_duration=hparams["avoid_if_longer_than"],
        min_duration=hparams["avoid_if_shorter_than"],
    )
