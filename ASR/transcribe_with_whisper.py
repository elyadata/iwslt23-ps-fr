#!/usr/bin/env python3
"""
    Recipe for fine-tuning a Whisper-Lin-transformer model for Pashto ASR.
"""
import json
import os
import sys
from typing import List

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from loguru import logger
from speechbrain.tokenizers.SentencePiece import SentencePiece

from train_asr_whisper import ASR


def export_json(transcription: List[dict], destination_file: str) -> None:
    with open(destination_file, 'wb') as file:
        json_transcriptions = json.dumps(transcription, ensure_ascii=False).encode('utf8')
        file.write(json_transcriptions)


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
        json_path=hparams["transcribe_dataset_json"],
        dynamic_items=[audio_pipeline, text_pipeline],
        output_keys=[
            "id",
            "sig",
            "start",
            "stop",
            "path",
            "duration",
            "tokens_bos",
            "tokens_eos",
        ],
    )

    if hparams["transcribe_dataset_json"] is not None:
        try:

            # Sorting the dataset in an ascending fashion
            dataset = dataset.filtered_sorted(sort_key="duration")

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
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Before training, we drop some of the Whisper Transformer Encoder layers
    if (
            len(asr_brain.modules.whisper.model.encoder.layers)
            > hparams["keep_n_layers"]
    ):
        asr_brain.modules.whisper.model.encoder.layers = asr_brain.modules.whisper.model.encoder.layers[
                                                         : hparams["keep_n_layers"]
                                                         ]

    else:
        n_last_layers_kept = hparams["keep_n_layers"]
        logger.warning(
            f"Cannot keep the {n_last_layers_kept} last layers of the Whisper encoder since it only has {len(asr_brain.modules.whisper.model.encoder.layers)} layers. Will not drop any layers."
        )

    # Load datasets for training, valid, and test, trains and applies tokenizer
    logger.info("Loading datasets...")
    dataset = dataio_prepare(hparams)

    logger.info("Loading pre-trained tokenizer...")
    pretrained_tokenizer = SentencePiece(
        model_dir=os.path.join(hparams["save_folder"]),
        vocab_size=hparams["vocab_size"],
        model_type="unigram"
    )

    logger.info("Generating transcriptions...")
    transcripts = asr_brain.transcribe_dataset(
        dataset=dataset,
        tokenizer=pretrained_tokenizer,
        min_key="WER",
        loader_kwargs=hparams['transcribe_dataloader_options']
    )

    logger.info("Saving transcriptions to a file...")
    if hparams["transcriptions_filename"] is not None:
        transcription_file = hparams["transcriptions_filename"] + ".json"
    else:
        transcription_file = "ontrac.st.unconstrained.contrastive1.json"
    export_json(transcripts, transcription_file)
    logger.success("Transcriptions saved successfully !")
