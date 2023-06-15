from elasticsearch import Elasticsearch
import tiktoken
from openai.embeddings_utils import get_embedding
from sentence_transformers import SentenceTransformer
from typing import List
import os
import yaml
config = yaml.safe_load(open("config.yml"))

# "text-search-ada-doc-001" # using text-search-ada-doc-001 instead of text-embedding-ada-002 becos of ES 1024 dims limit temporarily
# max_tokens = 1000  # the maximum for text-embedding-ada-002 is 8191

def parse_search_results(res: 'json', silent: bool = True) -> List[str]:
    document_ids = []
    for hit in res["hits"]["hits"]:
        document_ids.append(hit['_id'])
        if not silent:
            print(hit)
            print(f"Document ID: {hit['_id']}")
            print(f"Document Title: {hit['_source']['title']}")
            print(f"Document Desc: {hit['_source']['desc']}")
            print("==================================\n")

    return document_ids

def text_search(query: str, es_client: Elasticsearch,
    index: str, silent: bool = True) -> List[str]:

    query_dict = {    
        'query': {        
            'match': {            
                'title': query        
            }    
        }
    }
    res = es_client.search(index=index, body=query_dict, source=["title", "desc", "id"])

    if not silent:
        print("Query='{}'".format(query))

    return parse_search_results(res, silent)



def text_dis_max_search(query: str, es_client: Elasticsearch,
    index: str, silent: bool = True) -> List[str]:

    query_dict = {    
        'query': {        
            'dis_max': {
                'queries': [
                    {
                        'wildcard': {
                            'title': {
                                'value': '*' + query.strip() + '*',
                                'boost': 2
                                }
                            }
                    },
                    {
                        'wildcard': {
                            'desc': {
                                'value': '*' + query.strip() + '*',
                                'boost': 1
                                }
                            }
                    }
                    ]
                }
            }
        }

    res = es_client.search(index=index, body=query_dict,
                           source=["title", "desc", "id"])

    if not silent:
        print("Query='{}'".format(query))
        print("query_dict='{}'".format(query_dict))

    return parse_search_results(res, silent)

def ada_search(query: str, es_client: Elasticsearch,
    index: str, model_str: str = "text-search-ada-doc-001",
    top_k: int = 50, silent: bool = True) -> List[str]:

    query_vector = get_embedding(query, engine=model_str)
    query_dict = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": top_k
    }
    res = es_client.knn_search(index=index, knn=query_dict, source=["title", "desc", "id"])

    if not silent:
        print("Query='{}'".format(query))

    return parse_search_results(res, silent)


def ada_search_w_emb(query_vector: List[float], es_client: Elasticsearch,
    index: str, model_str: str = "text-search-ada-doc-001",
    top_k: int = 50, silent: bool = True) -> List[str]:

    query_dict = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": top_k
    }
    res = es_client.knn_search(index=index, knn=query_dict, source=["title", "desc", "id"])

    return parse_search_results(res, silent)

def sbert_search(query: str, es_client: Elasticsearch,
    index: str, model: SentenceTransformer,
    top_k: int = 50, silent: bool = True) -> List[str]:

    query_vector = [float(x) for x in model.encode(query)]
    query_dict = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": top_k
    }
    res = es_client.knn_search(index=index, knn=query_dict, source=["title", "desc", "id"])

    if not silent:
        print("Query='{}'".format(query))

    return parse_search_results(res, silent)


def combined_sbert_search(query: str, es_client: Elasticsearch,
    index: str, model: SentenceTransformer,
    top_k: int = 5, silent: bool = True) -> List[str]:

    query_vector = [float(x) for x in model.encode(query)]
    query_dict = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10
            },
        "query": {        
            "match": {            
                "title": query        
            }    
        }
    }
    res = es_client.search(index=index, body=query_dict, source=["title", "desc", "id"])

    if not silent:
        print("Query='{}'".format(query))

    return parse_search_results(res, silent)

def test_run(es_client: Elasticsearch, query_str = "smart cities") -> None:
    print("==================================Start of Text Search Results==================================\n\n\n")
    text_search(query=query_str,
        es_client=es_client,
        index=config['es']['pt_sbert_index'], silent=False)
    print("==================================End of Text Search Results==================================\n\n\n")

    print("==================================Start of Ada Search Results==================================\n\n\n")
    ada_search(query=query_str,
        es_client=es_client,
        index=config['es']['ada_index'],
        model_str=config['ada']['engine'], silent=False)
    print("==================================End of Ada Search Results==================================\n\n\n")

    print("==================================Start of Pre-trained SBert Search Results==================================\n\n\n")
    model = SentenceTransformer(config['sbert']['pretrained'])
    sbert_search(query=query_str,
        es_client=es_client,
        index=config['es']['pt_sbert_index'],
        model=model, silent=False)
    print("==================================End of Pre-trained SBert Search Results==================================\n\n\n")

    print("==================================Start of Fine-tuned SBert Search Results==================================\n\n\n")
    model = SentenceTransformer(os.path.join(config['search']['search_folder'], config['sbert']['finetuned']))
    sbert_search(query=query_str,
        es_client=es_client,
        index=config['es']['ft_sbert_index'],
        model=model, silent=False)
    print("==================================End of Fine-tuned SBert Search Results==================================\n\n\n")

    print("==================================Start of Combined Fine-tuned SBert Search Results==================================\n\n\n")
    model = SentenceTransformer(os.path.join(config['search']['search_folder'], config['sbert']['finetuned']))
    combined_sbert_search(query=query_str,
        es_client=es_client,
        index=config['es']['ft_sbert_index'],
        model=model, silent=False)
    print("==================================End of Combined Fine-tuned SBert Search Results==================================\n\n\n")
