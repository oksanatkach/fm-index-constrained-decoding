# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import logging
import multiprocessing
from functools import partial
import re

import ftfy
import torch
import tqdm

from transformers import AutoTokenizer

from index import FMIndex

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Global tokenizer in worker processes
_tokenizer = None


def init_worker(model_name):
    global _tokenizer
    if model_name is not None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        _tokenizer = torch.hub.load("pytorch/fairseq", "bart.large").eval()


def process(line, model_name):
    global _tokenizer

    if model_name is not None:
        is_bart = "bart" in model_name

        line = line.strip()
        if is_bart:
            line = " " + line
        with _tokenizer.as_target_tokenizer():
            return _tokenizer(line, add_special_tokens=False)["input_ids"] + [_tokenizer.eos_token_id]

    return _tokenizer.encode(" " + line.strip()).tolist()[1:]

def preprocess_file(input_path, labels, format="kilt", lowercase=False, tokenize=False):
    with open(input_path, "r", 2**16) as f:

        if format == "dpr":

            next(f)
            pieces_it = csv.reader(f, delimiter="\t", quotechar='"')
            pieces_it = ((pp[0], pp[2], pp[1]) for pp in pieces_it if len(pp) == 3)

        elif format == "kilt":

            pieces_it = (line.strip() for line in f)
            pieces_it = (line.split("\t", 2) for line in pieces_it)
            pieces_it = ((pp[0], pp[1], pp[2]) for pp in pieces_it if len(pp) == 3)

        pieces_it = tqdm.tqdm(pieces_it)

        for idx, title, text in pieces_it:

            idx = idx.strip()
            title = title.strip()

            text = re.sub(r"\s+", " ", text)
            text = ftfy.fix_text(text)
            text = text.replace("BULLET::::", "")
            text = text.replace("SECTION::::", "")
            text = text.strip()

            if not text:
                continue

            if tokenize:
                title = " ".join(word_tokenize(title))
                text = " ".join(word_tokenize(text))

            title = f"{title} {args.delim}"

            if args.include_title and title:
                text = f"{title} {text}"

            if lowercase:
                text = text.lower()

            labels.append(idx)

            yield text


def build_index(input_path):

    labels = []
    index = FMIndex()

    lines = preprocess_file(input_path, labels, args.format, lowercase=args.lowercase, tokenize=args.tokenize)

    process_func = partial(process, model_name=args.hf_model)
    with multiprocessing.Pool(args.jobs, initializer=init_worker, initargs=(args.hf_model,)) as p:
        sequences = p.imap(process_func, lines)
        index.initialize(sequences=sequences)

    index.labels = labels

    return index


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("input")
    # parser.add_argument("--input", default='data/alignment/tsvs/chosen.tsv')
    # parser.add_argument("--input", default='data/alignment/tsvs/rejected.tsv')
    # parser.add_argument("--input", default='data/PAQ/PAQ.tsv')
    # parser.add_argument("output")
    # parser.add_argument("--output", default='data/alignment/chosen_index/chosen.fm_index')
    # parser.add_argument("--output", default='data/alignment/rejected_index/rejected.fm_index')
    parser.add_argument("--output", default='data/PAQ/PAQ_index/PAQ.fm_index')
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include_title", action="store_true")
    parser.add_argument("--delim", default="@@")
    parser.add_argument("--format", choices=["kilt", "dpr"], default="kilt")
    # parser.add_argument("--hf_model", default=None, type=str)

    parser.add_argument("--hf_model", default="Qwen/Qwen3-8B", type=str)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print(args)

    if args.tokenize:
        from spacy.lang.en import English

        nlp = English()
        _tokenizer = nlp.tokenizer

        def word_tokenize(text):
            return [t.text.strip() for t in _tokenizer(text)]

    index = build_index(args.input)

    index.save(args.output)
