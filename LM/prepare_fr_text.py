#!/usr/bin/env python3
"""
Pashto-French data processing.

Author
------
Haroun Elleuch 2023
"""

import json
import os
import re
import sys
from typing import List, Union

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from loguru import logger
from tqdm import tqdm


def find_prepared_data(data_folder: str) -> bool:
    """This function will check if a `train.txt` and `dev.txt` files exist in the global 
    preprocessing output folder.

    Args:
        data_folder (str): Folder in which to check for existing files.

    Returns:
        bool: Existance of the files (True) or lack thereof (False).
    """
    if not os.path.exists(data_folder):
        return False
    if not os.path.isfile(
        os.path.join(data_folder, "train.txt")
    ) and not os.path.isfile(os.path.join(data_folder, "dev.txt")):
        return False
    return True


def get_folder_files(folders: List[str]) -> List[str]:
    """
    Given a list of folders containing (theoretically) text files, this function will iterate an return a list
    of the absolute paths for all the files.
    Args:
        folders: A list of folders.

    Returns:
        A list of files (absolute paths).
    """
    files = list()
    for path in folders:
        files += [
            os.path.join(path, entry)
            for entry in os.listdir(path)
            if os.path.isfile(os.path.join(path, entry))
        ]
    return files


def write_txt(txt_filename: str, data: List[str], mode="w"):
    """This function writes a list of strings into a file. By default, 
    will overwrite the specified output file but can also be set to append, or 
    to binary mode.

    Args:
        txt_filename (str): Name of the file to be created or updated.
        data (List[str]): Strings to be written to the output file.
        mode (str, optional): Writing mode. Defaults to "w": will overwrite any existing file with the same name.
    """
    logger.info(f"Writing file: {txt_filename}")
    with open(txt_filename, encoding="utf-8", mode=mode) as output_file:
        output_file.writelines(line + "\n" for line in data)

    logger.success(f"Saved: {txt_filename}")


def clean_text(input_text: str) -> Union[str, None]:
    """This function cleans the input text.

    Args:
        input_text (str): input text to be cleaned.

    Returns:
        str: Cleaned input text.
    """
    if ".txt" in input_text:
        return None
    # output = input_text.upper()
    output = input_text.replace("\n", "")
    output = re.sub("!+", "!", output)
    output = re.sub("-+", "-", output)
    output = re.sub("\?+", "?", output)
    output = re.sub(
        "\d+\t", "", output
    )  # removing line numbers from some files
    output = re.sub("/[…•·#^&*]/g", "", output)  # removing special characters
    output = output.replace("C?EST", "C'EST")
    output = output.strip()

    return output


def generate_dev_data(dev_json: str) -> List[str]:
    """
    This function is used to generate the `dev` split LM data.
    Args:
        dev_json (str): The `dev` JSON file containing the evaluation set transcriptions.

    Returns:
        List[str]: A list of sentences
    """
    logger.info("Generating dev french data for language model training...")
    output_text = list()

    # Loading speech segments descriptions of the split
    with open(dev_json) as file:
        description = json.load(file)

    for utt in tqdm(description):
        # Adding the translations to the LM text
        output_text.append(clean_text(utt["ref"]))

    return output_text


def generate_and_save_train_data(
    train_data_folder: str, output_file: str
) -> None:
    """
    This function is used to generate and combine train data into a single large file.
    Args:
        train_data_folder (str): Location of the training data files.
        output_file (str): File combining all preprocessed data.

    Returns:
        None
    """
    logger.info("Generating French train data for language model training...")
    for file in tqdm(get_folder_files(train_data_folder)):
        with open(file, "r") as f:
            lines = f.readlines()
            cleaned_lines = list()
            for line in tqdm(lines):
                cleaned_line = clean_text(line)
                if cleaned_line not in [
                    "",
                    "\n",
                    None,
                ]:
                    cleaned_lines.append(cleaned_line)
            write_txt(txt_filename=output_file, data=cleaned_lines, mode="a")
    return None


def data_proc(
    train_data_folder: str,
    dev_json_file: str,
    output_folder: str,
    skip: bool = False,
) -> None:
    """
    Prepare data for French LM training.
    Args:
        train_data_folder: Root folder of data (text file, only for now) to be used.
        dev_json_file: Dev split of the Pashto - FR dataset
        output_folder: Where to save the generated files.
        skip: Whether to skip or not the data pre-processing if the files are found.

    Returns:
        None
    """

    if skip and find_prepared_data(output_folder):
        logger.info(
            "Data files found and skipping enabled. Skipping data preparation."
        )
        return None

    # Create data output directory if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generating text data
    dev_data = generate_dev_data(dev_json_file)

    # Write files:
    generate_and_save_train_data(
        train_data_folder=train_data_folder,
        output_file=os.path.join(output_folder, "train.txt"),
    )
    write_txt(os.path.join(output_folder, "dev.txt"), dev_data)


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    data_proc(
        train_data_folder=hparams["train_data_folders"],
        dev_json_file=hparams["dev_data_json"],
        output_folder=hparams["data_folder"],
        skip=hparams["skip_preparation"],
    )
