from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
# from torch import nn
import os
import pandas as pd
import json
from common.utils import read_queries_res_as_df
import yaml
import logging
config = yaml.safe_load(open("config.yml"))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

###########################
#### Read Data  ####
###########################
queries_to_corpus = read_queries_res_as_df(config['data']['data_folder'])
print(queries_to_corpus.shape)
train_df=queries_to_corpus.sample(frac=0.8,random_state=200)
test_df=queries_to_corpus.drop(train_df.index)

# For evaluation purposes. Use only train dataset for fine-tuning
train_df[['query-id', 'corpus-id']].to_excel(os.path.join(config['data']['data_folder'], config['data']['qna_train']), index=False)
test_df[['query-id', 'corpus-id']].to_excel(os.path.join(config['data']['data_folder'], config['data']['qna_test']), index=False)

train_examples = []
for index, row in train_df.iterrows():
    try:
        query, paragraph = row['query'], row['paragraph']
        train_examples.append(InputExample(texts=[query, paragraph]))
    except:
        pass
print(len(train_examples))

# For the MultipleNegativesRankingLoss, it is important
# that the batch does not contain duplicate entries, i.e.
# no two equal queries and no two equal paragraphs.
# To ensure this, we use a special data loader
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)
print(len(train_dataloader.train_examples))

###########################
#### Defining Model  ####
###########################
logging.info("Defining Model")
# MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
# and trains the model so that is is suitable for semantic search
word_emb = models.Transformer('sentence-transformers/' + config['sbert']['pretrained'])
pooling = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pooling])
train_loss = losses.MultipleNegativesRankingLoss(model)


###########################
#### Fine Tuning Model  ####
###########################
logging.info("Fine Tuning Model")

# #Tune the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

logging.info("Saving Model")
os.makedirs(config['search']['search_folder'], exist_ok=True)
model.save(os.path.join(config['search']['search_folder'], config['sbert']['finetuned']))