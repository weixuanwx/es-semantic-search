import os
from elasticsearch import Elasticsearch
import pandas as pd
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def read_queries_res_as_df(data_folder: str = "data/") -> pd.DataFrame:
    '''
    Read the generated queries and responses into a pandas dataframe
    '''
    logging.info("Reading Query Data for Fine Tuning")
    with open(os.path.join(data_folder, "gen-3-queries.jsonl")) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    generated_queries = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    print(generated_queries.head())

    queries_to_corpus = pd.read_csv(os.path.join(data_folder, "gen-3-qrels/train.tsv"), sep='\t')
    print(queries_to_corpus.head())

    queries_to_corpus = queries_to_corpus.merge(generated_queries, left_on='query-id', right_on='_id', how='left')
    queries_to_corpus.rename(columns={'text':'query'}, inplace=True)
    queries_to_corpus.drop('_id', axis=1, inplace=True)
    print(queries_to_corpus.head())

    with open(os.path.join(data_folder, "corpus.jsonl")) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    corpus = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    corpus['paragraph'] = corpus['title'] + '; ' + corpus['text']
    corpus.rename(columns={'_id':'corpus-id'}, inplace=True)
    corpus['corpus-id'] = corpus['corpus-id'].astype(int)
    print(corpus.head())

    queries_to_corpus = queries_to_corpus.merge(corpus[['corpus-id','paragraph']], left_on='corpus-id', right_on='corpus-id', how='left')
    print(queries_to_corpus.head())
    print(queries_to_corpus.columns)
    print(queries_to_corpus.shape)

    return queries_to_corpus



def create_indices(es_client: 'Elasticsearch', index_name: str, n_dims: int = 1024):
    '''
    Creates a ElasticSearch index for ANN search.
    '''
    config = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "desc": {"type": "text"},
                "embedding": {
                        "type": "dense_vector",
                        "dims": n_dims,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    }

    #creates an index with a field of type dense_vector to allow nearest neighbor search
    if not es_client.indices.exists(index=[index_name]):
        es_client.indices.create(
            index=index_name,
            settings=config["settings"],
            mappings=config["mappings"],
        )
    else:
        logging.info("{} already exists".format(index_name))

    #check if the index has been created successfully
    logging.info(es_client.indices.exists(index=[index_name]))
    #True

def index_data(es_client: 'Elasticsearch', index_name: str, data_file: str, batch_size: int = 500):
    '''
    Indexes the excel data file into the designated index.
    '''

    logging.info("Indexing Data for index: {}".format(index_name))
    embedded_df = pd.read_csv(data_file)

    actions = []
    count = 0
    for index, row in embedded_df.iterrows():
        action = {"index": {"_index": index_name, "_id": row['id']}}
        doc = {
                "title": row['title'],
                "desc": row['desc'],
                "embedding": [float(x) for x in row['embedding'].strip('][').split(', ')]
            }
        
        actions.append(action)
        actions.append(doc)
        count += 1

        if count % batch_size == 0:
            es_client.bulk(index=index_name, operations=actions)
            actions = []
            logging.info("Indexed {} Documents for index: {}".format(count, index_name))

    if len(actions) > 0:
        es_client.bulk(index=index_name, operations=actions)

    result = es_client.count(index=index_name)

    logging.info("{} Documents found in index: {}".format(result.body['count'], index_name))