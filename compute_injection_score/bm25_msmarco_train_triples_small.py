import gzip
import os
import tarfile
import tqdm
import os
import json
from pyserini.index.lucene import IndexReader
from pyserini import analysis, search
from pyserini.pyclass import autoclass
import logging
from sentence_transformers import LoggingHandler, util
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

index_path = 'indexes/lucene-index-msmarco-passage' #lucene-index.cacm
index_reader = IndexReader(index_path)
b = 0.68
k1 = 0.82
similarity_bm25 = autoclass('org.apache.lucene.search.similarities.BM25Similarity')(k1, b)

scores = {}

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)


#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

train_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1')
if not os.path.exists(train_filepath):
    logging.info("Download queries.tar.gz")
    util.http_get('https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1', train_filepath)

with open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        pos_score, neg_score, qid, pos_id, neg_id = line.strip().split("\t")

        if qid not in scores.keys():
          scores[qid] = {}
        query = queries[qid]

        if pos_id in scores[qid].keys():
            pos_score = scores[qid][pos_id]
        else:
            pos_score = index_reader.compute_query_document_score(pos_id, query, similarity=similarity_bm25)
            scores[qid][pos_id] = pos_score
        
        neg_score = index_reader.compute_query_document_score(neg_id, query, similarity=similarity_bm25)
        scores[qid][neg_id] = neg_score

scores_dict_path = "bm25_scores_train_triples_small.json"
with open(scores_dict_path, "w+") as fp:
    json.dump(scores, indent=True, fp = fp)
