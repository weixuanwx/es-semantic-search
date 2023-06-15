
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel

import pathlib, os
import yaml
import logging
config = yaml.safe_load(open("config.yml"))
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

out_dir = config['data']['data_folder']
data_path = config['data']['data_folder']

#### Provide the data_path where scifact has been downloaded and unzipped
corpus = GenericDataLoader(data_path).load_corpus()
logging.info(corpus[list(corpus)[0]])


###########################
#### Query-Generation  ####
###########################

#### Model Loading 
model_path = config['query_gen']['beir_model']
generator = QGen(model=QGenModel(model_path))

#### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
#### https://huggingface.co/blog/how-to-generate
#### Prefix is required to seperate out synthetic queries and qrels from original
prefix = "gen-3"

#### Generating 3 questions per document for all documents in the corpus 
#### Reminder the higher value might produce diverse questions but also duplicates
ques_per_passage = 5

#### Generate queries per passage from docs in corpus and save them in original corpus
#### check your datasets folder to find the generated questions, you will find below:
#### 1. data/gen-3-queries.jsonl
#### 2. data/gen-3-qrels/train.tsv

batch_size = 64

generator.generate(corpus, output_dir=data_path, ques_per_passage=ques_per_passage, prefix=prefix, batch_size=batch_size)


# QUERY GENERATION FOR EVALUATION PURPOSES TOO