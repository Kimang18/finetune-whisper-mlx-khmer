from typing import List, Tuple, Dict, Iterator, Any
import argparse
import math
import time
from pathlib import Path
import sys
sys.setrecursionlimit(10000)
import itertools

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from khmernltk import word_tokenize
from mlx_whisper import transcribe
from mlx_whisper.tokenizer import Tokenizer
from mlx_whisper.whisper import Whisper

import evaluate
metric_wer = evaluate.load("wer")

from mlx.utils import tree_flatten


# Audio Feature Extractor
from mlx_whisper.audio import (
    N_FRAMES,
    N_SAMPLES,
    log_mel_spectrogram,
    pad_or_trim,
)

import tha.normalize
import tha.datetime
import tha.decimals
import tha.ordinals
import tha.currency

# Huggingface datasets
from datasets import load_dataset, Audio, concatenate_datasets, IterableDataset

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
        default=-1,
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
        default=16,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Adam learning rate."
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
    parser.add_argument("--seed", type=int, default=168, help="The PRNG seed")
    return parser


def load_google(args):
    hf_dataset = "google/fleurs"

    hf_dataset_lang = "km_kh"
    log.info(f"Loading dataset {hf_dataset}, {hf_dataset_lang} from hugging face")

    dataset = load_dataset(
        hf_dataset,
        hf_dataset_lang,
        trust_remote_code=True,
    )
    dataset = dataset.select_columns(["audio", "transcription"])
    dataset = dataset.map(transform_khmer_sentence)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset['train'], dataset['validation'], dataset['test']


def load_mix(args):
    dataset1 = load_dataset(
        "google/fleurs",
        "km_kh",
        trust_remote_code=True,
    )
    dataset1 = dataset1.select_columns(["audio", "transcription"])
    # dataset1 = dataset1.cast_column("audio", Audio(sampling_rate=16000))
    train, valid, test = dataset1["train"], dataset1["validation"], dataset1["test"]
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    # print(train.features)

    dataset2 = load_dataset("seanghay/km-speech-corpus")
    dataset2 = dataset2.select_columns(["audio", "transcription"])
    dataset2 = dataset2['train'].cast_column("audio", Audio(sampling_rate=16000))

    dataset3 = load_dataset("seanghay/khmer_mpwt_speech")
    dataset3 = dataset3.select_columns(["audio", "transcription"])
    dataset3 = dataset3['train'].cast_column("audio", Audio(sampling_rate=16000))

    train = concatenate_datasets([train, dataset2, dataset3])

    return train.to_iterable_dataset(), valid.to_iterable_dataset(), test.to_iterable_dataset()


def load_seanghay(args):
    dataset1 = load_dataset("seanghay/khmer_mpwt_speech")
    dataset1 = dataset1.select_columns(["audio", "transcription"])
    dataset1 = dataset1.map(transform_khmer_sentence)
    dataset1 = dataset1['train'].cast_column("audio", Audio(sampling_rate=16000))

    dataset2 = load_dataset("seanghay/km-speech-corpus")
    dataset2 = dataset2.select_columns(["audio", "transcription"])
    dataset2 = dataset2.map(transform_khmer_sentence)
    dataset2 = dataset2['train'].cast_column("audio", Audio(sampling_rate=16000))

    dataset2 = concatenate_datasets([dataset1, dataset2])
    dataset2 = dataset2.shuffle()
    # dataset2 = dataset2.train_test_split(test_size=0.2)
    # train, valid = dataset2['train'], dataset2['test']

    return dataset2, [-1]


def load_test(args):
    hf_dataset = "google/fleurs"
    hf_dataset_lang = "km_kh"
    log.info(f"Loading dataset {hf_dataset}, {hf_dataset_lang} from hugging face")
    dataset_g = load_dataset(
        hf_dataset,
        hf_dataset_lang,
        trust_remote_code=True,
        split="test"
    )
    dataset_g = dataset_g.map(transform_khmer_sentence)
    dataset_g = dataset_g.cast_column("audio", Audio(sampling_rate=16000))

    ds_openslr = load_dataset(
        "openslr",
        "SLR42",
        trust_remote_code=True,
        split="train"
    )
    ds_openslr = ds_openslr.rename_column("sentence", "transcription")
    ds_openslr = ds_openslr.map(transform_khmer_sentence)
    ds_openslr = ds_openslr.cast_column("audio", Audio(sampling_rate=16000))
    return dataset_g, ds_openslr


def transform_khmer_sentence(ds) -> Dict:
    transcription = tha.normalize.processor(ds["transcription"])
    l = [w for w in word_tokenize(transcription, return_tokens=True) if w != " "]
    for j in range(len(l)):
        w = l[j]
        if "$" in w or "៛" in w:
            w = tha.currency.processor(w)
        else:
            w = tha.datetime.time_processor(w)
            w = tha.datetime.date_processor(w)
            w = tha.decimals.processor(w)
            w = tha.ordinals.processor(w)
        l[j] = w.replace("▁", " ")
    transcription = " ".join(l)
    #transcription = word_tokenize(ds["transcription"], return_tokens=False, separator=" ")
    return {"transcription": transcription}

def iterate_batches(dset: IterableDataset, tokenizer: Tokenizer, batch_size: int, dtype=mx.float16, model_n_mels: int = 80, max_seq_length: int = 448, train: bool = False) -> Iterator[BatchInput]:
    # Shuffle indices
    while True:
        if train:
            # shuffled_dset: IterableDataset = dset.shuffle(
            #     seed=np.random.randint(0, 1000), buffer_size=20000)
            shuffled_dset = StatefuleIterableDataset(dset.shuffle(
                seed=np.random.randint(0, 1000), buffer_size=20000), batch_size)
        else:
            # shuffled_dset: IterableDataset = dset
            shuffled_dset = StatefuleIterableDataset(dset, batch_size)

        # Collect batches from dataset
        while True:
            # ds: List[Dict[str, Any]] = list(shuffled_dset.take(batch_size))
            ds: List[Dict[str, Any]] = list(itertools.islice(iter(shuffled_dset), batch_size))

            if len(ds) == 0:
                break
            batch_arr_mels: mx.array = get_array_mel_segments(
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
            # shuffled_dset = shuffled_dset.skip(batch_size)

        if not train:
            break

class StatefuleIterableDataset:
    def __init__(self, ds, batch_size=4):
        self.iterator = iter(ds)

    def __iter__(self):
        return self.iterator


def get_array_mel_segments(audio_arrs: List[np.array], n_mels: int, dtype) -> mx.array:
    batch_mel_segments: List[mx.array] = [
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


def get_array_tokens(texts: List[str], tokenizer: Tokenizer, max_seq_length: int) -> Tuple[mx.array, mx.array, mx.array]:
    batch_size: int = len(texts)
    batch: List[List[int]] = [
        [*tokenizer.sot_sequence_including_notimestamps] +
            tokenizer.encode(text)
        for text in texts]
    for x in batch:
        if x[-1] != tokenizer.eot:
            x.append(tokenizer.eot)
    batch_target: List[List[int]] = [x[1:] for x in batch]

    lengths: List[int] = [len(x) for x in batch]
    # Pad to the nearest multiple of 8 or the maximum length
    pad_to: int = 8
    max_length_in_batch: int = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
    max_length_in_batch = min(max_length_in_batch, max_seq_length)

    batch_arr: np.array = tokenizer.eot * np.ones(
        (batch_size, max_length_in_batch), np.int32)
    batch_arr_tars: np.array = tokenizer.eot * np.ones_like(batch_arr)
    for j in range(batch_size):
        truncated_length: int = min(lengths[j], max_seq_length)
        batch_arr[j, :truncated_length] = batch[j][:truncated_length]
        batch_arr_tars[j, :truncated_length-1] = batch_target[j][:truncated_length-1]
        lengths[j] = (truncated_length-1)

    # sanity check
    # special_tokens = tokenizer.special_tokens.values()
    # for x, xx, t in zip(batch_arr, batch_arr_tars, texts):
    #     print(tokenizer.decode([tok for tok in x if tok not in special_tokens]))
    #     print(tokenizer.decode([tok for tok in xx if tok not in special_tokens]))
    #     print(word_tokenize(t, separator=" ", return_tokens=False))

    batch_arr_text_tokens: mx.array = mx.array(batch_arr, dtype=mx.int32)
    batch_arr_target_tokens: mx.array = mx.array(batch_arr_tars, dtype=mx.int32)
    return batch_arr_text_tokens, batch_arr_target_tokens, mx.array(lengths)


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    references = []
    predictions = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size,
                        model_n_mels=model.dims.n_mels),
    ):
        # losses, ntoks = loss(model, *batch)
        losses, ntoks, l_list, t_list = loss_and_transcription(model, tokenizer, *batch)
        all_losses.append((losses * ntoks).item())
        ntokens += ntoks
        mx.eval(all_losses, ntokens)
        #all_wers.append(wers)
        predictions += l_list
        references += t_list

    # return np.sum(all_losses) / ntokens, np.mean(all_wers)
    wer = metric_wer.compute(references=references, predictions=predictions)
    return np.sum(all_losses) / ntokens, wer


def loss_and_transcription(model, tokenizer, mels, dec_input_tokens, target_tokens, token_lengths):
    # logits = model(mels, dec_input_tokens)
    # logits = logits.astype(mx.float32)
    # length_mask = mx.arange(target_tokens.shape[1])[
    #     None, :] < token_lengths[:, None]

    logits, length_mask = get_logits_and_length_mask(model, mels, dec_input_tokens, target_tokens, token_lengths)

    ce = nn.losses.cross_entropy(logits, target_tokens) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    length_mask = np.array(length_mask, dtype=bool)
    logits = np.array(mx.argmax(logits, axis=2), dtype=np.uint32)
    target_tokens = np.array(target_tokens, dtype=np.uint32)

    special_tokens = tokenizer.special_tokens.values()
    # print(tokenizer.decode(logits[0]))
    # l = [token for token in logits[0][length_mask[0]] if token not in special_tokens]
    # t = [token for token in target_tokens[0][length_mask[0]] if token not in special_tokens]
    # # print(len(t), len(target_tokens[0][length_mask[0]]))

    # l = tokenizer.decode(l)
    # # print("predi", l)
    # t = tokenizer.decode(t)
    # print("label", t)
    # print("*************************")

    l_list, t_list= [], []
    for lm, l, t in zip(length_mask, logits, target_tokens):
        # remove padding
        l = l[lm]
        t = t[lm]
        # remove special tokens
        l = [token for token in l if token not in special_tokens]
        t = [token for token in t if token not in special_tokens]
        # get transcription
        l = tokenizer.decode(l)
        t = tokenizer.decode(t)
        # print(l, "\n", t, "\n", 20*"-")
        l_list.append(l)
        t_list.append(t)
    #wer = metric_wer.compute(references=t_list, predictions=l_list)
    #return ce, ntoks, wer
    return ce, ntoks, l_list, t_list


def loss(model: Whisper, mels: mx.array, dec_input_tokens: mx.array, target_tokens: mx.array, token_lengths: mx.array) -> Tuple[mx.array, int]:
    # Run model on inputs
    # logits: mx.array = model(mels, dec_input_tokens)
    # logits = logits.astype(mx.float32)
    # length_mask: mx.array = mx.arange(target_tokens.shape[1])[
    #     None, :] < token_lengths[:, None]
    logits, length_mask = get_logits_and_length_mask(model, mels, dec_input_tokens, target_tokens, token_lengths)

    ce: mx.array = nn.losses.cross_entropy(logits, target_tokens) * length_mask
    ntoks: int = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def get_logits_and_length_mask(model: Whisper, mels: mx.array, dec_input_tokens: mx.array, target_tokens: mx.array, token_lengths: mx.array):
    # Run model on inputs
    logits: mx.array = model(mels, dec_input_tokens)
    logits = logits.astype(mx.float32)
    length_mask: mx.array = mx.arange(target_tokens.shape[1])[
        None, :] < token_lengths[:, None]
    return logits, length_mask


# def train(model, train_set, val_set, loss, tokenizer, args, num_iters, val_num_iters):
def train(model, train_set, loss, tokenizer, args, num_iters):
    log.info("Training")
    # optimizer = optim.Adam(learning_rate=args.learning_rate)
    warmup_steps = 0.05*num_iters
    decay_steps2 = 0.1*num_iters
    decay_steps1 = num_iters - warmup_steps - decay_steps2
    warmup = optim.linear_schedule(0.0, args.learning_rate, steps=warmup_steps)
    # lin_decay = optim.linear_schedule(args.learning_rate, 0.0, steps=decay_steps)
    lin_decay1 = optim.cosine_decay(args.learning_rate, end=1e-5, decay_steps=decay_steps1)
    lin_decay2 = optim.cosine_decay(1e-5, end=0.0, decay_steps=decay_steps2)
    scheduler = optim.join_schedules([warmup, lin_decay1, lin_decay2], [warmup_steps, decay_steps1+warmup_steps])

    weight_decay = 0.1 # 0.00221 # 0.1
    adam_epsilon = 1e-9
    optimizer = optim.AdamW(learning_rate=scheduler,
                            betas=(0.89, 0.79), # (0.88, ...)
                            eps=adam_epsilon,
                            weight_decay=weight_decay,
                            bias_correction=False)

    dtype = mx.float16 if eval(args.fp16) else mx.float32

    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0
    start = time.perf_counter()

    # Main training loop
    for it, batch in zip(
        range(num_iters),
        iterate_batches(train_set, tokenizer, args.batch_size, dtype=dtype,
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
            # val_loss, wer = evaluate(
            #     model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            # )
            # log.info(
            #     f"Iter {it + 1}: "
            #     f"Val loss {val_loss:.3f}, "
            #     f"Val wer {wer:.3f}, "
            #     f"Val took {(time.perf_counter() - stop):.3f}s"
            # )
            transcribe("./tests/voice5.mp3", model=model,
                       verbose=True, fp16=eval(args.fp16))
            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            log.info(
                f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")

        # Final validation loss if needed
        if (it + 1) == num_iters:
            stop = time.perf_counter()
            # val_loss, wer = evaluate(
            #     model, val_set, loss, tokenizer, args.batch_size, val_num_iters
            # )
            # log.info(
            #     f"Iter {it + 1}: "
            #     f"Val loss {val_loss:.3f}, "
            #     f"Val wer {wer:.3f}, "
            #     f"Val took {(time.perf_counter() - stop):.3f}s"
            # )
            transcribe("./tests/voice5.mp3", model=model,
                       verbose=True, fp16=eval(args.fp16))
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {}
    if args.train:
        tokenizer_config["language"] = "km"  # args.language
    dtype = mx.float16 if eval(args.fp16.title()) else mx.float32
    model, tokenizer, config = lora_utils.load(
        args.model, dtype, tokenizer_config)
    log.info(f"tokenizer language {tokenizer.language}")

    # # Freeze all layers & create LORA layers
    model.freeze()
    lora_utils.linear_to_lora(model, args.lora_layers, rank=64, target_modules=['query', 'key', 'value', 'out'], cross_target_modules=['query', 'key', 'value', 'out'], alpha=64, dropout=0.02858)

    # Load dataset
    log.info("Loading datasets")
    # load_google(args)  # load(args)
    # train_set, val_set, test_set = load_mix(args)
    # train_set, val_set, test_set = load_google(args)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        log.info(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        train_set, val_set = load_seanghay(args)
        num_iters = int(train_set.num_rows/ args.batch_size)
        # val_num_iters = int(val_set.num_rows/ args.batch_size)
        # Train model
        # train(model, train_set.to_iterable_dataset(), val_set.to_iterable_dataset(), loss, tokenizer, args, args.iters, val_num_iters)
        train(model, train_set.to_iterable_dataset(), loss, tokenizer, args, args.iters)

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
        test_set1, test_set2 = load_test(args)

        test_num_iters = int(test_set1.num_rows/ args.batch_size)
        model.eval()
        test_loss, test_wer = evaluate(
            model,
            test_set1.to_iterable_dataset(),
            loss,
            tokenizer,
            args.batch_size,
            num_batches=test_num_iters,
        )
        test_ppl = math.exp(test_loss)

        log.info(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
        log.info(f"Google Test wer {test_wer:.3f}.")

        test_num_iters = int(test_set2.num_rows/ args.batch_size)
        test_loss, test_wer = evaluate(
            model,
            test_set2.to_iterable_dataset(),
            loss,
            tokenizer,
            args.batch_size,
            num_batches=test_num_iters,
        )
        test_ppl = math.exp(test_loss)

        log.info(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
        log.info(f"OPENSLR Test wer {test_wer:.3f}.")
