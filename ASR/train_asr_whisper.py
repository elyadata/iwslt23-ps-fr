#!/usr/bin/env python3
"""
    Recipe for fine-tuning a Whisper-Lin-transformer model for Pashto ASR.
"""

import logging
import sys

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        tokens_bos, _ = batch.tokens_bos  # transcriptions

        # Whisper module
        feats = self.modules.whisper(wavs)

        # dimensionality reduction
        src = self.modules.enc(feats)

        # transformer decoder
        if self.distributed_launch:
            dec_out = self.modules.Transformer.module.forward_mt_decoder_only(
                src, tokens_bos, pad_idx=self.hparams.pad_index
            )
        else:
            dec_out = self.modules.Transformer.forward_mt_decoder_only(
                src, tokens_bos, pad_idx=self.hparams.pad_index
            )

        # logits and softmax
        pred = self.modules.seq_lin(dec_out)
        p_seq = self.hparams.log_softmax(pred)

        # compute outputs
        hyps = None
        if stage == sb.Stage.VALID:
            # the output of the encoder (enc) is used for valid search
            hyps, _ = self.hparams.valid_search(src.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(src.detach(), wav_lens)

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_seq, _, hyps) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # st loss
        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if loss.isnan():
            logger.warning(f"NaN loss found: {loss}")

        fr_detokenizer = MosesDetokenizer(lang=self.hparams.lang)

        if stage != sb.Stage.TRAIN:
            predictions = [
                fr_detokenizer.detokenize(
                    tokenizer.sp.decode_ids(utt_seq).split(" ")
                )
                for utt_seq in hyps
            ]

            detokenized_transcription = [
                fr_detokenizer.detokenize(transcription.split(" "))
                for transcription in batch.transcription
            ]
            # it needs to be a list of list due to the extend on the bleu implementation
            # targets = [detokenized_transcription]
            
            # tracking error rate
            self.wer_metric.append(batch.id, predictions, detokenized_transcription)
            self.cer_metric.append(batch.id, predictions, detokenized_transcription)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def init_optimizers(self):
        """ Initializes the whisper optimizer if the model is not whisper_frozen """
        if not self.hparams.whisper_frozen:
            self.whisper_optimizer = self.hparams.whisper_opt_class(
                self.modules.whisper.parameters()
            )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def zero_grad(self, set_to_none=False):
        if not self.hparams.whisper_frozen:
            self.whisper_optimizer.zero_grad(set_to_none)
        self.adam_optimizer.zero_grad(set_to_none)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()

        if self.check_gradients(loss):
            if not self.hparams.whisper_frozen:  # if Whisper is not frozen
                self.whisper_optimizer.step()
            self.adam_optimizer.step()

        if not self.hparams.whisper_frozen:
            self.whisper_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer() 
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["ACC"] = self.acc_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["WER"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            if not self.hparams.whisper_frozen:
                (
                    old_lr_whisper,
                    new_lr_whisper,
                ) = self.hparams.lr_annealing_whisper(stage_stats["WER"])
                sb.nnet.schedulers.update_learning_rate(
                    self.whisper_optimizer, new_lr_whisper
                )
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch": current_epoch,
                        "lr_adam": old_lr_adam,
                        "lr_whisper": old_lr_whisper,
                    },
                    train_stats={"loss": self.train_stats},
                    valid_stats=stage_stats,
                )
            else:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": current_epoch, "lr_adam": old_lr_adam},
                    train_stats={"loss": self.train_stats},
                    valid_stats=stage_stats,
                )

            # create checkpoint
            meta = {"WER": stage_stats["WER"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, min_keys=["WER"]
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the filename 
    # and a start and stop timestamps from the dict passed as argument
    @sb.utils.data_pipeline.takes("path", "start", "stop", "id")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, id):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""

        sig = sb.dataio.dataio.read_audio({"start": start, "stop": stop, "file": wav})
        torch.cuda.empty_cache()
        return sig

    @sb.utils.data_pipeline.takes("path", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav, start, stop):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio({"start": start, "stop": stop, "file": wav})
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription", "tokens_list", "tokens_bos", "tokens_eos"
    )
    def reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        tokens_list = tokenizer.sp.encode_as_ids(translation)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    data_folder = hparams["data_folder"]

    # 1. train tokenizer on the data
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["vocab_size"],
        annotation_train=data_folder + "/train.json",
        annotation_read="transcription",
        annotation_format="json",
        model_type="unigram",
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )

    # 2. load data and tokenize with trained tokenizer
    datasets = {}
    for dataset in ["train", "dev"]:
        json_path = f"{data_folder}/{dataset}.json"

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline_func, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "transcription",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
            ],
        )

    for dataset in ["dev", "test"]:
        json_path = f"{data_folder}/{dataset}.json"
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "transcription",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["dev"] = datasets["dev"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["dev"] = datasets["dev"].filtered_sorted(
                sort_key="duration"
            )

        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["dev"] = datasets["dev"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["dev"] = datasets["dev"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_debug_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
            )
            datasets["dev"] = datasets["dev"].filtered_sorted(
                key_min_value={"duration": hparams["sorting_min_duration"]},
                key_max_value={"duration": hparams["sorting_max_duration"]},
                sort_key="duration",
            )

        hparams["dataloader_options"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # creates a logger
    logger = logging.getLogger(__name__)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create main experiment class
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Data preparation
    import prepare_ps_asr

    run_on_main(
        prepare_ps_asr.data_proc,
        kwargs={
            "dataset_folder": hparams["root_data_folder"],
            "output_folder": hparams["data_folder"],
            "max_duration": hparams["max_segment_duration"],
        },
    )
    # Load datasets for training, valid, and test, trains and applies tokenizer
    datasets, tokenizer = dataio_prepare(hparams)

    # Before training, we drop some of the Whisper Transformer Encoder layers
    if len(asr_brain.modules.whisper.model.encoder.layers) > hparams["keep_n_layers"]:
        asr_brain.modules.whisper.model.encoder.layers = asr_brain.modules.whisper.model.encoder.layers[
            : hparams["keep_n_layers"]
        ]
    n_last_layers_kept = hparams["keep_n_layers"]
    logger.warning(f"Cannot keep the {n_last_layers_kept} last layers of the Whisper encoder since it only has {len(asr_brain.modules.whisper.model.encoder.layers)} layers. Will not drop any layers.")

    # Training
    asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            datasets["train"],
            datasets["dev"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["test_dataloader_options"],
        )

    # Test    
    logger.info("Evaluating last checkpoint:")
    for dataset in ["dev", "test"]:
        asr_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )

    logger.info("Evaluating best checkpoint (least WER):")
    for dataset in ["dev", "test"]:
        test_stats = asr_brain.evaluate(
            test_set=datasets[dataset],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_options"],
        )