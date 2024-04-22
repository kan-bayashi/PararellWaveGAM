#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import resampy
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.datasets import AudioSCPDataset
from parallel_wavegan.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py)."
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file. if use, you must to specify both scp and segments.",
    )
    parser.add_argument(
        "--text",
        default=None,
        type=str,
        help="kaldi-style text format hubert embedding index.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help="kaldi-style utt2spk file. If you want to add global conditionning with "
        "speaker id, you need to specify this argument.",
    )
    parser.add_argument(
        "--spk2idx",
        default=None,
        type=str,
        help="kaldi-style spk2idx file. If you want to add global conditionning with "
        "speaker id, you need to specify this argument.",
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help="directory including wav files. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    print("=====args.rootdir",args.rootdir)
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # get text
    print("text filr", args.text)
    print("spk2idx", args.spk2idx)
    
    with open(args.text) as f:
        lines = [line.strip() for line in f.readlines()]
    text = {line.split(" ")[0]: line.split(" ")[1:-1] for line in lines}

    # load spk2utt file
    if args.utt2spk is not None:
        with open(args.utt2spk) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        utt2spk = {l.split()[0]: l.split()[1] for l in lines}
        if args.spk2idx is not None:
            with open(args.spk2idx) as f:
                lines = [l.replace("\n", "") for l in f.readlines()]
            spk2idx = {l.split()[0]: int(l.split()[1]) for l in lines}

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        if utt_id not in text.keys():
            continue
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # downsample
        if fs != config["sampling_rate"]:
            audio = resampy.resample(audio, fs, config["sampling_rate"], axis=0)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        # use hubert index instead of mel
        mel = np.array(text[utt_id]).astype(np.int64).reshape(-1, 1)

        if args.spk2idx is not None:
            spk = utt2spk[utt_id]
            if spk in spk2idx:
                idx = spk2idx[spk]
            else:
                logging.warn(f"{spk} is unknown speaker.")
                max_idx = max(spk2idx.values()) + 1
                idx = max_idx

            # concatenate with mel
            idx = np.repeat(np.array(idx).reshape(1, 1), len(mel), axis=0)
            mel = np.concatenate([mel, idx], axis=1)

        # make sure the audio length and feature length are matched
        logging.info(f"Mod: {len(audio) - len(mel) * config['hop_size']}")
        assert len(mel) * config["hop_size"] <= len(audio), f"{utt_id}: {len(mel)}, {config['hop_size']}, {len(audio)}"

        audio = audio[: len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(
                f"{utt_id} causes clipping. "
                f"it is better to re-consider global gain scale."
            )
            continue

        # save
        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "feats",
                mel.astype(np.float32),
            )
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                mel.astype(np.float32),
                allow_pickle=False,
            )
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()
