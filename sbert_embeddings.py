import pandas as pd, numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer

import yaml
import logging
config = yaml.safe_load(open("config.yml"))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

raw_data_path = os.path.join(config['data']['data_folder'], config['data']['raw_data'])
pretrained_emb_path = os.path.join(config['data']['data_folder'], config['data']['data_w_pt_sbert_emb'])
finetrained_emb_path = os.path.join(config['data']['data_folder'], config['data']['data_w_ft_sbert_emb'])

try:
    logging.info("Reading {}...".format(config['data']['data_w_pt_sbert_emb']))
    pretrained_emb_df = pd.read_csv(pretrained_emb_path)
except FileNotFoundError:
    logging.info("FileNotFoundError - will encode and save")
    logging.info("Encoding...")
    data_df = pd.read_csv(raw_data_path)
    model = SentenceTransformer(config['sbert']['pretrained'])

    encoded_data = model.encode(data_df['combined'].tolist())
    # encoded_data = np.asarray(encoded_data.astype('float32'))
    data_df["embedding"] = list(encoded_data)
    data_df["embedding"] = data_df["embedding"].apply(lambda emb: [float(x) for x in emb])
    logging.info(data_df["embedding"].values[0])

    cols = ["id", "title", "desc", "combined", "embedding"]
    data_df[cols].to_csv(pretrained_emb_path, index=False)

try:
    logging.info("Reading {}...".format(config['data']['data_w_ft_sbert_emb']))
    finetuned_emb_df = pd.read_csv(finetrained_emb_path)
except FileNotFoundError:
    logging.info("FileNotFoundError - will encode and save")
    logging.info("Encoding...")
    data_df = pd.read_csv(raw_data_path)
    model = SentenceTransformer(os.path.join(config['search']['search_folder'], config['sbert']['finetuned']))

    encoded_data = model.encode(data_df['combined'].tolist())
    # encoded_data = np.asarray(encoded_data.astype('float32'))
    data_df["embedding"] = list(encoded_data)
    data_df["embedding"] = data_df["embedding"].apply(lambda emb: [float(x) for x in emb])

    logging.info("Saving...")
    cols = ["id", "title", "desc", "combined", "embedding"]
    data_df[cols].to_csv(finetrained_emb_path, index=False)