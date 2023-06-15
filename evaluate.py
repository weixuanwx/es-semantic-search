import os
import json
import time
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from common.utils import read_queries_res_as_df
from search import text_search, ada_search_w_emb, sbert_search, text_dis_max_search, combined_sbert_search, test_run
from sentence_transformers import SentenceTransformer
from openai.embeddings_utils import get_embedding
import yaml
import logging
config = yaml.safe_load(open("config.yml"))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

###########################
#### Label the results batch  ####
###########################
batch = config['evaluate']['batch']


###########################
#### Read Data  ####
###########################
logging.info("Run Query and Responses Data")
queries_to_corpus = read_queries_res_as_df(config['data']['data_folder'])
print(queries_to_corpus.shape)


###########################
#### Batch Encode Queries for Ada embeddings first ####
###########################
logging.info("Batch Encode Queries for Ada embeddings first")
trg_query_ada_emb_path = os.path.join(config['data']['data_folder'], config['data']['trg_query_ada_emb'])
print(trg_query_ada_emb_path)
to_encode = False
new = False
try:
    ada_query_emb_df = pd.read_csv(trg_query_ada_emb_path)

    # We are using only a subsample of the entire corpus for Ada. Due to limited usage credits.
    ada_subsample_df = pd.read_csv(os.path.join(config['data']['data_folder'], config['data']['data_w_ada_emb']))
    to_encode_df = queries_to_corpus[queries_to_corpus['corpus-id'].isin(ada_subsample_df['id'])]
    not_encoded_df = to_encode_df[~to_encode_df['query-id'].isin(ada_query_emb_df['query-id'])]

    not_encoded = len(not_encoded_df)
    if not_encoded > 0:
        logging.info("{} queries not encoded. To encode.".format(not_encoded))
        to_encode = True
except FileNotFoundError:
    to_encode = True
    new = True

if to_encode:
    logging.info("Queries without Ada embeddings found. Encode first")
    if new:
        ada_query_emb_df = pd.DataFrame()
    else:
        ada_query_emb_df = pd.read_csv(trg_query_ada_emb_path)
    query_ids, queries, vectors_list = [], [], []
    count, step = 0, 100
    for query_id, query in zip(not_encoded_df['query-id'], not_encoded_df['query']):
        if len(ada_query_emb_df) > 0 and query_id in set(ada_query_emb_df['query-id']):
            continue
        query_vector = get_embedding(query, engine=config['ada']['engine'])
        query_ids.append(query_id); queries.append(query); vectors_list.append(query_vector)
        count += 1
        if count % step == 0:
            tmp_df = pd.DataFrame({'query-id': query_ids, 'query': queries, 'vectors': vectors_list})
            ada_query_emb_df = pd.concat([ada_query_emb_df, tmp_df])
            ada_query_emb_df.to_csv(trg_query_ada_emb_path, index=False)
            query_ids, queries, vectors_list = [], [], []
            logging.info("Encoded {} queries".format(len(ada_query_emb_df)))
        time.sleep(3)
    tmp_df = pd.DataFrame({'query-id': query_ids, 'query': queries, 'vectors': vectors_list})
    ada_query_emb_df = pd.concat([ada_query_emb_df, tmp_df])
    ada_query_emb_df.to_csv(trg_query_ada_emb_path, index=False)

###########################
#### Run searches for each query  ####
###########################
logging.info("Run searches for each query")

# Test Run the search functions
logging.info("Authenticate ES")
es_client = Elasticsearch(config['es']['url'], 
                http_auth=(config['es']['user'], config['es']['password']),
                verify_certs=config['es']['cert'])
pt_sbert_model = SentenceTransformer(config['sbert']['pretrained'])
ft_sbert_model = SentenceTransformer(os.path.join(config['search']['search_folder'], config['sbert']['finetuned']))

# Test Run the search functions
test_run(es_client=es_client)

# read latest queries with ada embeddings
# # full run
# ada_query_emb_df = pd.read_csv(trg_query_ada_emb_path)
# queries_df = queries_to_corpus.merge(ada_query_emb_df[['query-id','vectors']], how='left')
# queries_df["vectors"] = queries_df["vectors"].apply(lambda emb: [float(x) for x in emb.strip('][').split(', ')] if pd.notnull(emb) else None)
# # run only on those with ada embeddings
queries_df = pd.read_csv(trg_query_ada_emb_path)
queries_df["vectors"] = queries_df["vectors"].apply(lambda emb: [float(x) for x in emb.strip('][').split(', ')] if pd.notnull(emb) else None)

# read train and test set
train_df = pd.read_excel(os.path.join(config['data']['data_folder'], config['data']['qna_train']))
train_df['uid'] = train_df['query-id'].astype(str) + '-' + train_df['corpus-id'].astype(str)
test_df = pd.read_excel(os.path.join(config['data']['data_folder'], config['data']['qna_test']))
test_df['uid'] = test_df['query-id'].astype(str) + '-' + test_df['corpus-id'].astype(str)

# Run searches and collect results
# # check for existing results
results_path = os.path.join(config['evaluate']['result_folder'], "results_{}.xlsx".format(batch))
to_evaluate = False
try:
    results_df = pd.read_excel(results_path)
    not_evaluated_queries = queries_df[~queries_df['query-id'].isin(results_df['query-id'])]['query-id'].to_list()
    if len(not_evaluated_queries) > 0:
        logging.info("{} queries not evaluated. To continue evaluating...".format(len(not_evaluated_queries)))
        to_evaluate = True
except FileNotFoundError:
    to_evaluate = True
    new = True

if to_evaluate:
    if not os.path.exists(config['evaluate']['result_folder']):
        os.makedirs(config['evaluate']['result_folder'])

    if new:
        results_df = pd.DataFrame()
    else:
        results_df = pd.read_excel(results_path)
    count, step = 0, 100

    query_ids, queries, response_ids = [], [], []
    text_match, text2_match, ada_match, pt_sbert_match, ft_sbert_match, c_ft_sbert_match = [], [], [], [], [], []
    text_returns, text2_returns, ada_returns, pt_sbert_returns, ft_sbert_returns, c_ft_sbert_returns = [], [], [], [], [], []
    for query_id, query, ada_query_vectors in zip(queries_df['query-id'], queries_df['query'], queries_df['vectors']):
        print(query)
        response_id = int(queries_to_corpus[queries_to_corpus['query-id']==query_id]['corpus-id'].values[0])
        query_ids.append(query_id); queries.append(query); response_ids.append(response_id)

        text_res_ids = text_search(query=query, es_client=es_client, index=config['es']['pt_sbert_index']); text_res_ids = [int(s) for s in text_res_ids]; text_returns.append(text_res_ids)
        text2_res_ids = text_dis_max_search(query=query, es_client=es_client, index=config['es']['pt_sbert_index']); text2_res_ids = [int(s) for s in text2_res_ids]; text2_returns.append(text2_res_ids)
        if ada_query_vectors:
            ada_res_ids = ada_search_w_emb(query_vector=ada_query_vectors, es_client=es_client, index=config['es']['ada_index'], model_str=config['ada']['engine']); ada_res_ids = [int(s) for s in ada_res_ids]; ada_returns.append(ada_res_ids)
        else:
            ada_returns.append(None)
        pt_sbert_res_ids = sbert_search(query=query, es_client=es_client, index=config['es']['pt_sbert_index'], model=pt_sbert_model); pt_sbert_res_ids = [int(s) for s in pt_sbert_res_ids]; pt_sbert_returns.append(pt_sbert_res_ids)
        ft_sbert_res_ids = sbert_search(query=query, es_client=es_client, index=config['es']['ft_sbert_index'], model=ft_sbert_model); ft_sbert_res_ids = [int(s) for s in ft_sbert_res_ids]; ft_sbert_returns.append(ft_sbert_res_ids)
        comb_ft_sbert_res_ids = combined_sbert_search(query=query, es_client=es_client, index=config['es']['ft_sbert_index'], model=ft_sbert_model); comb_ft_sbert_res_ids = [int(s) for s in comb_ft_sbert_res_ids]; c_ft_sbert_returns.append(comb_ft_sbert_res_ids)

        # MRR of Position matches
        try:
            text_match.append( 1.0/(text_res_ids.index(response_id)+1.0) )
        except ValueError:
            text_match.append(0.0)
        try:
            text2_match.append( 1.0/(text2_res_ids.index(response_id)+1.0) )
        except ValueError:
            text2_match.append(0.0)
        if ada_query_vectors:
            try:
                ada_match.append( 1.0/(ada_res_ids.index(response_id)+1.0) )
            except ValueError:
                ada_match.append(0.0)
        else:
            ada_match.append(None)
        try:
            pt_sbert_match.append( 1.0/(pt_sbert_res_ids.index(response_id)+1.0) )
        except ValueError:
            pt_sbert_match.append(0.0)
        try:
            ft_sbert_match.append( 1.0/(ft_sbert_res_ids.index(response_id)+1.0) )
        except ValueError:
            ft_sbert_match.append(0.0)
        try:
            c_ft_sbert_match.append( 1.0/(comb_ft_sbert_res_ids.index(response_id)+1.0) )
        except ValueError:
            c_ft_sbert_match.append(0.0)

        count += 1
        if count % step == 0:
            print(len(c_ft_sbert_match))
            print(len(c_ft_sbert_returns))
            tmp_df = pd.DataFrame({'query-id': query_ids, 'query': queries, 'response_id': response_ids,
                'text_rr': text_match, 'text2_rr': text2_match, 'ada_rr': ada_match, 'pt_sbert_rr': pt_sbert_match, 'ft_sbert_rr': ft_sbert_match, 'c_ft_sbert_rr': c_ft_sbert_match,
                'text_full_res': text_returns, 'text2_full_res': text2_returns, 'ada_full_res': ada_returns, 'pt_sbert_full_res': pt_sbert_returns, 'ft_sbert_full_res': ft_sbert_returns, 'c_ft_sbert_full_res': c_ft_sbert_returns
                })
            results_df = pd.concat([results_df, tmp_df])
            results_df['uid'] = results_df['query-id'].astype(str) + '-' + results_df['response_id'].astype(str)
            results_df['test_set'] = np.where(results_df['uid'].isin(test_df['uid']), 1, 0)
            results_df.to_excel(results_path, index=False)
            query_ids, queries, response_ids = [], [], []
            text_match, text2_match, ada_match, pt_sbert_match, ft_sbert_match = [], [], [], [], []
            text_returns, text2_returns, ada_returns, pt_sbert_returns, ft_sbert_returns = [], [], [], [], []
            logging.info("Evaluated {} queries".format(len(results_df)))

    tmp_df = pd.DataFrame({'query-id': query_ids, 'query': queries, 'response_id': response_ids,
        'text_rr': text_match, 'text2_rr': text2_match, 'ada_rr': ada_match, 'pt_sbert_rr': pt_sbert_match, 'ft_sbert_rr': ft_sbert_match, 'c_ft_sbert_rr': c_ft_sbert_match,
        'text_full_res': text_returns, 'text2_full_res': text2_returns, 'ada_full_res': ada_returns, 'pt_sbert_full_res': pt_sbert_returns, 'ft_sbert_full_res': ft_sbert_returns, 'c_ft_sbert_full_res': c_ft_sbert_returns
        })
    results_df = pd.concat([results_df, tmp_df])
    
    results_df['uid'] = results_df['query-id'].astype(str) + '-' + results_df['response_id'].astype(str)
    results_df['test_set'] = np.where(results_df['uid'].isin(test_df['uid']), 1, 0)
    results_df.to_excel(results_path, index=False)

