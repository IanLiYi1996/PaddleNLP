# coding=UTF-8

from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import hnswlib
from data import convert_example_test, create_dataloader
from data import gen_id2corpus, gen_text_file
from ann_util import build_index

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--similar_text_pair_file", type=str, required=True, help="The full path of similar text pair file")
parser.add_argument("--recall_result_dir", type=str, default='recall_result', help="The full path of recall result file to save")
parser.add_argument("--recall_result_file", type=str, default='recall_result_file', help="The file name of recall result")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=None, type=int, help="output_embedding_size")
parser.add_argument("--recall_num", default=10, type=int, help="Recall number for each query from Ann index.")

parser.add_argument("--hnsw_m", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=100, type=int, help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000, type=int, help="Recall number for each query from Ann index.")

parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    id2corpus = gen_id2corpus(args.corpus_file)

    corpus_list = [{idx: text} for idx, text in id2corpus.items()]

    # Tokenize input texts
    texts = [
        "There's a kid on a skateboard.",
        "A kid is skateboarding.",
        "A kid is inside the house."
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

