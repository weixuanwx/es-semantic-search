import os
from elasticsearch import Elasticsearch
from common.utils import create_indices, index_data
import pandas as pd
import yaml
config = yaml.safe_load(open("config.yml"))

es_client = Elasticsearch(config['es']['url'], 
                http_auth=(config['es']['user'], config['es']['password']),
                verify_certs=config['es']['cert'])

index_name = config['es']['pt_sbert_index']
if not es_client.indices.exists(index=[index_name]):
    create_indices(es_client, index_name, n_dims = config['sbert']['pretrained_dim'])
    index_data(es_client, index_name, os.path.join(config['data']['data_folder'], config['data']['data_w_pt_sbert_emb']))

index_name = config['es']['ft_sbert_index']
if not es_client.indices.exists(index=[index_name]):
    create_indices(es_client, index_name, n_dims = config['sbert']['finetuned_dim'])
    index_data(es_client, index_name, os.path.join(config['data']['data_folder'], config['data']['data_w_ft_sbert_emb']))

index_name = config['es']['ada_index']
if not es_client.indices.exists(index=[index_name]):
    create_indices(es_client, index_name, n_dims = config['ada']['ada_dim'])
    index_data(es_client, index_name, os.path.join(config['data']['data_folder'], config['data']['data_w_ada_emb']))
