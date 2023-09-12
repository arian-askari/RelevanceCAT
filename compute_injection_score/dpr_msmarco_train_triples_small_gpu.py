import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import os, tarfile, tqdm, os, json
from sentence_transformers import SentenceTransformer, util
from pyserini.index.lucene import IndexReader
from pyserini import analysis, search
from pyserini.pyclass import autoclass
import logging
from sentence_transformers import LoggingHandler, util

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout
# Load the model
model = SentenceTransformer(
    "sentence-transformers/msmarco-distilbert-base-tas-b", device="cuda"
)
scores_dict = {}
### Now we read the MS Marco dataset
data_folder = "msmarco-data"
os.makedirs(data_folder, exist_ok=True)
#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get(
            "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
            tar_filepath,
        )
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(collection_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage
### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get(
            "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
            tar_filepath,
        )
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(queries_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query
train_filepath = os.path.join(
    data_folder, "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1"
)
if not os.path.exists(train_filepath):
    logging.info("Download queries.tar.gz")
    util.http_get(
        "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        train_filepath,
    )
teachers_scores_dict = (
    {}
)  # we do this so we can compute scores for query in batch and it be more efficient!
with open(train_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        pos_score, neg_score, qid, pos_id, neg_id = line.strip().split("\t")
        if qid not in teachers_scores_dict:
            teachers_scores_dict[qid] = {}
        teachers_scores_dict[qid][pos_id] = pos_score
        teachers_scores_dict[qid][neg_id] = neg_score
print("len(teachers_scores_dict):", len(teachers_scores_dict.keys()))
print("computing dpr scores!")
for qid, did_scores in tqdm.tqdm(
    teachers_scores_dict.items(),
    unit_scale=False,
    desc="computing dpr scores per query...",
):
    query = queries[qid]
    dids = did_scores.keys()
    docs = [corpus[did] for did in dids]
    # Encode query and documents
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    # Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    # Combine docs & scores
    doc_score_pairs = list(zip(dids, scores))
    # #Sort by decreasing score
    # doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    scores_dict[qid] = {}
    for did, score in doc_score_pairs:
        # print(score, did)
        scores_dict[qid][did] = score
scores_dict_path = "1_dpr_scores_train_triples_small_gpu.json"
with open(scores_dict_path, "w+") as fp:
    json.dump(scores_dict, indent=True, fp=fp)
