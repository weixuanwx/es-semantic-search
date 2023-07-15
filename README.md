# Content

## Introduction

- This project will attempt to build a semantic search service while comparing the results from two language models: OpenAI's Ada Embeddings and Sentence-BERT
- Because we only tag 1 correct document to each query, the evaluation scores are not fully indicative of the performance of the models. They are only used as a general comparison since we don't have fully labelled data. Close inspection of the results are required

## Pre-installation

- [Install Docker Hub](https://docs.docker.com/docker-hub/quickstart/)

## Setup Docker Images for ElasticSearch service, Kibana service and finetuning language models

### ElasticSearch service container

- `docker pull docker.elastic.co/elasticsearch/elasticsearch:8.7.0`
- `docker network create elastic`
- `docker run --name es01 --net elastic -p 9200:9200 -it docker.elastic.co/elasticsearch/elasticsearch:8.7.0`
- `docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .`

### Kibana service container

- `docker pull docker.elastic.co/kibana/kibana:8.7.0`
- `docker run --name kib-01 --net elastic -p 5601:5601 docker.elastic.co/kibana/kibana:8.7.0`
- Click to the link `http://0.0.0.0:5601/?code=<yourKibanaCode>`
- Enter the enrollment token from the output of the ElasticSearch service

### To remove the ES and Kibana containers and their network, run:

- `docker network rm elastic`
- `docker rm es01`
- `docker rm kib-01`

### Build or import Docker Image for fine tuning SBert

- Builds a Docker Image for finetuning and encoding sbert. Also, uses this image for indexing data.
- `docker build -t search-sbert .`

## Query your data

- Your python script for querying data
- `mkdir data/`
- `python query_data.py`
- Data needs to have the following columns: ["id", "title", "desc"]

## Query generation to create pseudo training and test data set

- Uses the Beir library to automatically generate search queries for each document
- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 query_generation.py`

## Finetune sbert model

- Splits the queries and responses into train and test set.
- Finetunes the model with the training set.
- Best to run the finetuning with a GPU or it gets very slow.
- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 fine_tune_sbert.py`

## Encode dataset with embeddings

### Ada embeddings

#### Get Open AI token and add to environment variables

- Following OpenAI's best practices [2] for API safety
- Log in to your OpenAI account > click on your profile photo > click on 'View API keys'
- Click 'Create new secret key' and copy down your API key
- For Linux / MacOS:
    1. Setup your API key in your environment: `echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc`
    2. Update your shell: `source ~/.zshrc`

#### Run the Ada embeddings script

- Uses the `text-search-ada-doc-001` model for now as ElasticSearch, at the time of writing this, is unable to run ANN search on documents with more than 1024 dimensions
- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 ada_embeddings.py`

### SBert embeddings

#### Run the SBert embeddings script

- Uses the pre-trained sbert model and your newly finetuned model to separately encode your dataset
- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 sbert_embeddings.py`

## Create and index the embeddings in ElasticSearch

- Index the data into your local ElasticSearch index. Feel free to configure the URL and security settings.
- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 indexing.py`
- Currently the script indexes the data into separate indexes, one for each type of embeddings. They could also be combined into one index.

## Benchmark

### Run the evaluation script

- `docker run -it --rm -v "$(pwd):/run" -w /run --name query_gen search-sbert python3 evaluate.py`

### Evaluation metrics

- For evaluation purposes in the offline setting, since we do not have actual click-through rates, we will be unable to use Clickthrough rate.
- Also, since we only know if 1 document is relevant from the set of documents returned, we will not use Normalized Discounted Cumulative Gain (NDCG), Precision@k, Recall@k
- We will use a proxy of Mean Reciprocal Rate (MRR) instead. However, this is not the true MRR because we do not have ground-truth labels for all responses for each query. We are just assuming only 1 response is correct for each query.

### Enhancements

- Improve on the vector model
- Combine the vector search with other search criteria such as filters or text similarity. E.g. filter documents by metatags

## Reference

1. [Implementing Nearest Neighbours Search with ElasticSearch](https://betterprogramming.pub/implementing-nearest-neighbour-search-with-elasticsearch-c59a8d33dd9d)
2. [Semantic Search with SBert](https://medium.com/mlearning-ai/semantic-search-with-s-bert-is-all-you-need-951bc710e160)
3. [OpenAI API key safety best practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)