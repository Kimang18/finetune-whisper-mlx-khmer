from typing import List, Tuple
import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from khmernltk import word_tokenize
from mlx_whisper import transcribe

# import evaluate
# metric_wer = evaluate.load("wer")

from mlx.utils import tree_flatten


# Audio Feature Extractor
from mlx_whisper.audio import (
    N_FRAMES,
    N_SAMPLES,
    log_mel_spectrogram,
    pad_or_trim,
)


# Huggingface datasets
from datasets import load_dataset, Audio, concatenate_datasets

# Configure typealias for batched inputs
from collections import namedtuple

import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


BatchInput = namedtuple(
    "BatchInput", "input_features dec_input_tokens target_tokens token_lengths")
issubclass(BatchInput, tuple)


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_whisper_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--fp16",
        default=True,
        help="Load data type float 16",
    )
    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int,
                        default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


def load_google(args):
    hf_dataset = "google/fleurs"

    hf_dataset_lang = "km_kh"
    log.info(f"Loading dataset {hf_dataset}, {
             hf_dataset_lang} from hugging face")

    dataset = load_dataset(
        hf_dataset,
        hf_dataset_lang,
        trust_remote_code=True,
    )
    dataset = dataset.select_columns(["audio", "transcription"])
    train, valid, test = dataset["train"].to_iterable_dataset(
    ), dataset["validation"].to_iterable_dataset(), dataset["test"].to_iterable_dataset()

    return train, valid, test


def load(args):
    # dataset2 = load_dataset("seanghay/khmer_mpwt_speech")
    # dataset2 = dataset2.select_columns(["audio", "transcription"])
    # dataset2 = dataset2.cast_column("audio", Audio(sampling_rate=16000))

    dataset2 = load_dataset("seanghay/km-speech-corpus")
    dataset2 = dataset2.select_columns(["audio", "transcription"])

    # dataset2 = concatenate_datasets([dataset1['train'], dataset2['train']])

    dataset2 = dataset2['train'].train_test_split(test_size=0.2)
    train, valid, test = dataset2['train'].to_iterable_dataset(
    ), dataset2['test'].to_iterable_dataset(), [-1]

    return train, valid, test


def iterate_batches(dset, tokenizer, batch_size, dtype=mx.float16, model_n_mels=80, max_seq_length=448, train=False):
    # Shuffle indices
    while True:
        if train:
            shuffled_dset = dset.shuffle(
                seed=np.random.randint(0, 1000), buffer_size=10000)
        else:
            shuffled_dset = dset

        # Collect batches from dataset
        while True:
            ds = list(shuffled_dset.take(batch_size))
            if len(ds) == 0:
                break
            batch_arr_mels = get_array_mel_segments(
                [ds[j]['audio']['array'] for j in range(len(ds))],
                model_n_mels,
                dtype
            )

            batch_arr_text_tokens, batch_arr_target_tokens, batch_token_lengths = get_array_tokens(
                [ds[j]["transcription"] for j in range(len(ds))],
                tokenizer,
                max_seq_length
            )

            yield BatchInput(input_features=batch_arr_mels,
                             dec_input_tokens=batch_arr_text_tokens,
                             target_tokens=batch_arr_target_tokens,
                             token_lengths=batch_token_lengths,
                             )
            if len(ds) < batch_size:  # whole pass over dataset
                break
            shuffled_dset = shuffled_dset.skip(batch_size)

        if not train:
            break


def get_array_mel_segments(audio_arrs: List[np.array], n_mels: int, dtype) -> mx.array:
    batch_mel_segments = [
        pad_or_trim(
            log_mel_spectrogram(
                audio_arr, n_mels=n_mels, padding=N_SAMPLES),
            N_FRAMES,
            axis=-2).astype(dtype)
        for audio_arr in audio_arrs
    ]
    if batch_mel_segments[0].shape[0] > 3000:
        print(
            "[WARNING] Some mel segments are longer than 3000 samples. "
            "Consider pre-splitting your data to save memory."
        )
    np_dtype = np.float16 if dtype == mx.float16 else np.float32
    return mx.array(
        np.array(batch_mel_segments, dtype=np_dtype),
        dtype=dtype
    )


def get_array_tokens(texts: List[str], tokenizer, max_seq_length) -> Tuple[mx.array, mx.array]:
    batch_size = len(texts)
    batch = [
        [*tokenizer.sot_sequence_including_notimestamps] +
        tokenizer.encode(word_tokenize(
            text, separator=" ", return_tokens=False))
        for text in texts]
    batch_target = [x[1:] + [tokenizer.eot] for x in batch]
    lengths = [len(x) for x in batch]

    # Pad to the nearest multiple of 8 or the maximum length
    pad_to = 8
    max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
    max_length_in_batch = min(max_length_in_batch, max_seq_length)

    batch_arr = tokenizer.eot * np.ones(
        (batch_size, max_length_in_batch), np.int32)
    batch_arr_tars = -100 * np.ones_like(batch_arr)
    for j in range(batch_size):
        truncated_length = min(lengths[j], max_seq_length)
        batch_arr[j, :truncated_length] = batch[j][:truncated_length]
        batch_arr_tars[j, :truncated_length] = batch_target[j][:truncated_length]
        lengths[j] = (truncated_length)

    batch_arr_text_tokens = mx.array(batch_arr, dtype=mx.int32)
    batch_arr_target_tokens = mx.array(batch_arr_tars, dtype=mx.int32)
    return batch_arr_text_tokens, batch_arr_target_tokens, mx.array(lengths)


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size,
                        model_n_mels=model.dims.n_mels),
    ):
        losses, ntoks = loss(model, *batch)
        all_losses.append((losses * ntoks).item())
        ntokens += ntoks
        mx.eval(all_losses, ntokens)

    return np.sum(all_losses) / ntokens


def loss(model, mels, dec_input_tokens, target_tokens, token_lengths):
    # Run model on inputs
    logits = model(mels, dec_input_tokens)
    logits = logits.astype(mx.float32)
    length_mask = mx.arange(target_tokens.shape[1])[
        None, :] < token_lengths[:, None]

    ce = nn.losses.cross_entropy(logits, target_tokens) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def train(model, train_set, val_set, loss, tokenizer, args):
    log.info("Training")
    # optimizer = optim.Adam(learning_rate=args.learning_rate)
    weight_decay = 0.01
    adam_epsilon = 1e-8
    optimizer = optim.AdamW(learning_rate=args.learning_rate,
                            eps=adam_epsilon,
                            weight_decay=weight_decay,
                            bias_correction=True)

    dtype = mx.float16 if args.fp16 else mx.float32

    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0
    start = time.perf_counter()

    # Main training loop
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size,
                        model_n_mels=model.dims.n_mels, train=True),
    ):
        # print(it, [*batch][0].shape, [*batch][1].shape, [*batch][2].shape)
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            log.info(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"Tokens/sec {n_tokens / (stop - start):.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            log.info(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
            transcribe("./tests/voice5.mp3", model=model,
                       fp16=args.fp16, verbose=True)
            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            log.info(
                f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {}
    if args.train:
        tokenizer_config["language"] = "km"  # args.language
    dtype = mx.float16 if args.fp16 else mx.float32
    model, tokenizer, config = lora_utils.load(
        args.model, dtype, tokenizer_config)
    log.info(f"tokenizer language {tokenizer.language}")

    # # Freeze all layers & create LORA layers
    model.freeze()
    lora_utils.linear_to_lora(model, args.lora_layers)

    # Load dataset
    log.info("Loading datasets")
    # load_google(args)  # load(args)
    train_set, val_set, test_set = load(args)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        log.info(f"Loading pretrained adapters from {
                 args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        # Train model
        train(model, train_set, val_set, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **
                 dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )

    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        log.info("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)

        log.info(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
        log.info(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
