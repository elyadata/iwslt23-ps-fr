#!/usr/bin/env python3

import json
import os
import sys
from typing import List

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from loguru import logger
from speechbrain.tokenizers.SentencePiece import SentencePiece

from train_st_whisper_scraped_data import ST


def export_json(transcription: List[dict], destination_file: str) -> None:
    with open(destination_file, "wb") as file:
        json_transcriptions = json.dumps(
            transcription, ensure_ascii=False
        ).encode("utf8")
        file.write(json_transcriptions)


def export_text(translations: List[dict], destination_file: str) -> None:
    lines = list()
    for elem in translations:
        lines.append(elem['transcription'].replace('&', '')+'\n')
    with open(destination_file, "w") as file:
        file.writelines(lines)

# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the filename
    # and a start and stop timestamps from the dict passed as argument
    @sb.utils.data_pipeline.takes("path", "start", "stop", "id")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(path, start, stop, id):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(
            {"start": start, "stop": stop, "file": path}
        )
        return sig

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos")
    def text_pipeline(id):
        """Generate BOS and EOS tokens for transcribing"""
        tokens_bos = torch.LongTensor([hparams["bos_index"]])
        yield tokens_bos
        tokens_eos = torch.LongTensor([hparams["eos_index"]])
        yield tokens_eos

    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["translate_dataset_json"],
        dynamic_items=[audio_pipeline, text_pipeline],
        output_keys=[
            "id",
            "sig",
            "start",
            "stop",
            "path",
            "tokens_bos",
            "tokens_eos",
        ],
    )

    if hparams["translate_dataset_json"] is not None:
        try:
            hparams["dataloader_options"]["shuffle"] = False
            return dataset
        except FileNotFoundError as e:
            print("Cannot create a transcription dataset: ", e)
    else:
        print("No manifest for dataset transcription was specified.")

    return None


if __name__ == "__main__":

    logger.info("Loading parameters...")
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    logger.info("Instantiating model...")
    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Before training, we drop some of the Whisper Transformer Encoder layers
    if (
            len(st_brain.modules.whisper.model.encoder.layers)
            > hparams["keep_n_layers"]
    ):
        st_brain.modules.whisper.model.encoder.layers = st_brain.modules.whisper.model.encoder.layers[
                                                        : hparams["keep_n_layers"]]

    else:
        n_last_layers_kept = hparams["keep_n_layers"]
        logger.warning(
            f"Cannot keep the {n_last_layers_kept} last layers of the Whisper encoder since it only has \
            {len(st_brain.modules.whisper.model.encoder.layers)} layers. Will not drop any layers."
        )

    # Load dataset
    logger.info("Loading datasets...")
    dataset = dataio_prepare(hparams)

    logger.info("Loading pre-trained tokenizer...")
    pretrained_tokenizer = SentencePiece(
        model_dir=os.path.join(hparams["save_folder"]),
        vocab_size=hparams["vocab_size"],
        model_type="unigram",
    )

    logger.debug(f"Dataset: {dataset}")

    logger.info("Generating transcriptions...")
    transcripts = st_brain.translate_dataset(
        dataset=dataset,
        tokenizer=pretrained_tokenizer,
        max_key="BLEU",
        loader_kwargs=hparams["transcribe_dataloader_options"],
    )

    transcription_file = hparams["translation_file_output"]
    logger.info(f"Saving transcriptions to {transcription_file}")
    export_json(transcripts, transcription_file+".json")
    export_text(transcripts, transcription_file+".txt")
    logger.success("Transcriptions saved successfully !")
